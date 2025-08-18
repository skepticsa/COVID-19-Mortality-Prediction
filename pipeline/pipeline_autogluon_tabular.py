"""
SageMaker Pipelines DAG for AutoGluon Tabular (JumpStart built-in; single-instance training).

Flow:
  1) ProcessingStep (transform.py): feature engineering -> engineered.csv (headered)
  2) ProcessingStep (preprocess.py): split + format data for JumpStart (no header, label first)
  3) TrainingStep (JumpStart AutoGluon Tabular)
  4) ProcessingStep (evaluate.py): offline eval on val set
  5) ConditionStep (classification): accuracy >= threshold
  6) RegisterModel (only on pass)
"""

import os
from typing import Optional

import boto3
import sagemaker
from sagemaker import image_uris, model_uris, script_uris, hyperparameters
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.session import Session
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import JsonGet, Join
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel


def get_pipeline(
    region: Optional[str] = None,
    role_arn: Optional[str] = None,
    default_bucket: Optional[str] = None,
    pipeline_name: str = "Autogluon-Tabular-Pipeline",
) -> Pipeline:
    """Create the SageMaker Pipeline for AutoGluon Tabular (classification)."""
    boto_sess = boto3.Session(region_name=region) if region else boto3.Session()
    sagemaker_sess = Session(boto_session=boto_sess)
    region = sagemaker_sess.boto_region_name
    pipeline_sess = PipelineSession(boto_session=boto_sess)
    role = role_arn or sagemaker.get_execution_role(sagemaker_session=sagemaker_sess)
    default_bucket = default_bucket or sagemaker_sess.default_bucket()

    cache_config = CacheConfig(enable_caching=True, expire_after="30d")

    # --------- Parameters ----------
    input_data_s3 = ParameterString(
        name="InputDataS3Uri",
        default_value=f"s3://{default_bucket}/autogluon/input/covid.csv",
    )
    label_col = ParameterString(name="LabelColumn", default_value="label")  # pass "mortality" at start
    # Retained for compatibility; steps hard-code classification
    problem_type = ParameterString(name="ProblemType", default_value="classification")

    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.2xlarge")

    # CPU default is safer for compile-time container resolution; override at start if you have GPU quota.
    train_instance_type = ParameterString(name="TrainInstanceType", default_value="ml.m5.4xlarge")

    time_limit = ParameterInteger(name="TimeLimitSeconds", default_value=900)  # AutoGluon budget, in seconds
    auto_stack = ParameterString(name="AutoStack", default_value="false")
    presets = ParameterString(name="Presets", default_value="medium_quality_faster_train")
    val_fraction = ParameterFloat(name="ValFraction", default_value=0.2)

    acc_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.80)
    # Unused in classification but retained so your run script still works unchanged
    rmse_threshold = ParameterFloat(name="RMSEThreshold", default_value=1e9)

    model_package_group = ParameterString(name="ModelPackageGroupName", default_value="AutogluonTabular")
    approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")

    # --------- Step 0: Feature engineering ----------
    fe_script = os.path.join("scripts", "transform.py")
    fe_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        sagemaker_session=pipeline_sess,
        base_job_name="autogluon-fe",
    )

    step_fe = ProcessingStep(
        name="FeatureEngineer",
        processor=fe_processor,
        code=fe_script,
        job_arguments=[
            "--input-s3-uri", input_data_s3,
            "--label-name", label_col,  # e.g., "mortality"
            "--elderly-age", "65",
            "--delay-threshold-days", "2",
            "--output-filename", "engineered.csv",
        ],
        inputs=[
            ProcessingInput(
                input_name="raw",
                source=input_data_s3,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(output_name="engineered", source="/opt/ml/processing/engineered"),
            ProcessingOutput(output_name="meta",       source="/opt/ml/processing/meta"),
        ],
        cache_config=cache_config,
    )

    # --------- Step 1: Preprocess ----------
    preprocess_script = os.path.join("scripts", "preprocess.py")
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=role,
        instance_type=processing_instance_type,
        instance_count=1,
        sagemaker_session=pipeline_sess,
        base_job_name="autogluon-preprocess",
    )

    step_process = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        code=preprocess_script,
        job_arguments=[
            "--input-s3-uri",
            Join(
                on="/",
                values=[
                    step_fe.properties.ProcessingOutputConfig.Outputs["engineered"].S3Output.S3Uri,
                    "engineered.csv",
                ],
            ),
            "--label-column", label_col,          # pass "mortality" when you start the pipeline
            "--problem-type", "classification",
            "--val-fraction", val_fraction.to_string(),
        ],
        inputs=[
            ProcessingInput(
                input_name="engineered",
                source=step_fe.properties.ProcessingOutputConfig.Outputs["engineered"].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            )
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="val",   source="/opt/ml/processing/val"),
            ProcessingOutput(output_name="eval",  source="/opt/ml/processing/eval"),
            ProcessingOutput(output_name="meta",  source="/opt/ml/processing/meta"),
        ],
        cache_config=cache_config,
    )

    # --------- Step 2: Train (JumpStart AutoGluon Tabular) ----------
    train_model_id = "autogluon-classification-ensemble"
    infer_model_id = "autogluon-classification-ensemble"

    # Your env requires framework=... (older SDK); newer SDKs will ignore this harmlessly.
    train_image_uri = image_uris.retrieve(
        framework="autogluon",
        region=region,
        model_id=train_model_id,
        model_version="*",
        image_scope="training",
        instance_type=train_instance_type,  # resolved to default_value at compile time
    )
    train_source_uri = script_uris.retrieve(model_id=train_model_id, model_version="*", script_scope="training")
    train_model_uri = model_uris.retrieve(model_id=train_model_id, model_version="*", model_scope="training")

    hps = hyperparameters.retrieve_default(model_id=train_model_id, model_version="*")
    hps["time_limit"] = time_limit.to_string()    # numeric param -> to_string()
    hps["auto_stack"] = auto_stack                # ParameterString is fine
    hps["presets"] = presets                      # ParameterString is fine

    estimator = Estimator(
        role=role,
        image_uri=train_image_uri,
        source_dir=train_source_uri,
        model_uri=train_model_uri,
        entry_point="transfer_learning.py",
        instance_count=1,               # JumpStart AutoGluon is single-instance
        instance_type=train_instance_type,
        max_run=86400,                  # generous job budget so AutoGluon time_limit controls stop
        hyperparameters=hps,
        output_path=f"s3://{default_bucket}/autogluon/output",
        sagemaker_session=pipeline_sess,
        enable_network_isolation=False,
        disable_profiler=True,          # avoids cache misses in Pipelines
    )

    step_train = TrainingStep(
        name="TrainAutogluon",
        estimator=estimator,
        inputs={
            "training": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,
    )

    # --------- Step 3: Evaluate ----------
    eval_processor = ScriptProcessor(
        image_uri=train_image_uri,   # ensures AutoGluon is available inside the container
        role=role,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        sagemaker_session=pipeline_sess,
        base_job_name="autogluon-eval",
    )

    evaluate_script = os.path.join("scripts", "evaluate.py")
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")

    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=eval_processor,
        code=evaluate_script,
        job_arguments=[
            "--model-artifacts-s3", step_train.properties.ModelArtifacts.S3ModelArtifacts,
            "--eval-data-s3",
            Join(
                on="/",
                values=[
                    step_process.properties.ProcessingOutputConfig.Outputs["eval"].S3Output.S3Uri,
                    "val_with_header.csv",
                ],
            ),
            "--label-column", label_col,
            "--problem-type", "classification",
        ],
        inputs=[
            ProcessingInput(
                input_name="model_artifacts",
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                input_name="eval_data",
                source=step_process.properties.ProcessingOutputConfig.Outputs["eval"].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        property_files=[evaluation_report],
        cache_config=cache_config,
    )

    # --------- Step 4: Gate + Register (classification only) ----------
    acc_json = JsonGet(step_name=step_eval.name, property_file=evaluation_report, json_path="primary_metric.value")

    # Resolve a CPU inference image (portable default for model registry)
    infer_image_uri = image_uris.retrieve(
        framework="autogluon",
        region=region,
        model_id=infer_model_id,
        model_version="*",
        image_scope="inference",
        instance_type="ml.m5.large",
    )

    eval_s3_uri = Join(
        on="/",
        values=[step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri, "evaluation.json"],
    )
    model_metrics = ModelMetrics(model_statistics=MetricsSource(s3_uri=eval_s3_uri, content_type="application/json"))

    # IMPORTANT: do NOT pass sagemaker_session here when using estimator=...
    register_step_cls = RegisterModel(
        name="RegisterModelClassification",
        estimator=estimator,  # ensures compiler path has a non-None estimator
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        image_uri=infer_image_uri,  # override training image with inference image
        content_types=["text/csv"],
        response_types=["text/csv", "application/json"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge", "ml.g5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group,
        approval_status=approval_status,
        model_metrics=model_metrics,
    )

    fail_step = FailStep(
        name="FailIfQualityLow",
        error_message="Model quality did not meet the threshold.",
    )

    cond_cls_ok = ConditionStep(
        name="CheckAccuracyThreshold",
        conditions=[ConditionGreaterThanOrEqualTo(left=acc_json, right=acc_threshold)],
        if_steps=[register_step_cls],
        else_steps=[fail_step],
    )

    # --------- Assemble ----------
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_data_s3,
            label_col,
            problem_type,           # retained for compatibility with your run script
            processing_instance_type,
            train_instance_type,
            time_limit,
            auto_stack,
            presets,
            val_fraction,
            acc_threshold,
            rmse_threshold,         # retained (unused) for compatibility
            model_package_group,
            approval_status,
        ],
        steps=[step_fe, step_process, step_train, step_eval, cond_cls_ok],
        sagemaker_session=pipeline_sess,
    )
    return pipeline


if __name__ == "__main__":
    region = os.environ.get("AWS_DEFAULT_REGION")
    pipe = get_pipeline(region=region)
    print(f"Pipeline '{pipe.name}' created. Use .upsert() and .start() from a Notebook or run run_pipeline.py.")
