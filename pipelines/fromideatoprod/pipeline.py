
import pandas as pd
import json
import boto3
import pathlib
import io
import os
import sagemaker
import mlflow
from time import gmtime, strftime, sleep
from sagemaker.deserializers import CSVDeserializer
from sagemaker.serializers import CSVSerializer

from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.processing import (
    ProcessingInput, 
    ProcessingOutput, 
    ScriptProcessor
)
from sagemaker.inputs import TrainingInput

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep, 
    TrainingStep, 
    CreateModelStep,
    CacheConfig
)
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.parameters import (
    ParameterInteger, 
    ParameterFloat, 
    ParameterString, 
    ParameterBoolean
)
from sagemaker.workflow.quality_check_step import (
    DataQualityCheckConfig,
    ModelQualityCheckConfig,
    QualityCheckStep,
)
from sagemaker.workflow.clarify_check_step import (
    ModelBiasCheckConfig, 
    ClarifyCheckStep, 
    ModelExplainabilityCheckConfig
)
from sagemaker import Model
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.conditions import (
    ConditionGreaterThan,
    ConditionGreaterThanOrEqualTo
)
from sagemaker.workflow.parallelism_config import ParallelismConfiguration
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import (
    Join,
    JsonGet
)
from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)
from sagemaker.lambda_helper import Lambda

from sagemaker.model_metrics import (
    MetricsSource, 
    ModelMetrics, 
    FileSource
)
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig 
from sagemaker.image_uris import retrieve
from sagemaker.workflow.function_step import step
from sagemaker.workflow.step_outputs import get_step
from sagemaker.model_monitor import DatasetFormat, model_monitoring

from pipelines.preprocess import preprocess
from pipelines.evaluate import evaluate
from pipelines.register import register
from pipelines.extract import prepare_datasets

def get_sagemaker_client(region):
     return boto3.Session(region_name=region).client("sagemaker")

def get_pipeline_session(region, bucket_name):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        bucket_name: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=bucket_name,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_name=None):
    try:
        print(f"Getting project tags for {sagemaker_project_name}")
        
        sm_client = get_sagemaker_client(region)
        
        project_arn = sm_client.describe_project(ProjectName=sagemaker_project_name)['ProjectArn']
        project_tags = sm_client.list_tags(ResourceArn=project_arn)['Tags']

        print(f"Project tags: {project_tags}")
        
        for project_tag in project_tags:
            new_tags.append(project_tag)
            
    except Exception as e:
        print(f"Error getting project tags: {e}")
        
    return new_tags
    
def get_pipeline(
    region,
    sagemaker_project_id=None,
    sagemaker_project_name=None,
    role=None,
    bucket_name=None,
    bucket_prefix="",
    input_s3_url=None,
    feature_group_name=None,
    model_package_group_name="from-idea-to-prod-model-group",
    pipeline_name_prefix="from-idea-to-prod-pipeline",
    process_instance_type="ml.m5.large",
    train_instance_type="ml.m5.xlarge",
    test_score_threshold=0.70,
    tracking_server_arn=None,
):
    """Gets a SageMaker ML Pipeline instance.
    
    Returns:
        an instance of a pipeline
    """
    if feature_group_name is None and input_s3_url is None:
        print("One of feature_group_name or input_s3_url must be provided. Exiting...")
        return None

    session = get_pipeline_session(region, bucket_name)
    sm = session.sagemaker_client
    
    if role is None:
        role = sagemaker.session.get_execution_role(session)

    print(f"sagemaker version: {sagemaker.__version__}")
    print(f"Execution role: {role}")
    print(f"Input S3 URL: {input_s3_url}")
    print(f"Feature group: {feature_group_name}")
    print(f"Model package group: {model_package_group_name}")
    print(f"Pipeline name prefix: {pipeline_name_prefix}")
    print(f"Tracking server ARN: {tracking_server_arn}")
    
    pipeline_name = f"{pipeline_name_prefix}-{sagemaker_project_id}"
    experiment_name = pipeline_name

    output_s3_prefix = f"s3://{bucket_name}/{bucket_prefix}"
    # Set the output S3 url for model artifact
    output_s3_url = f"{output_s3_prefix}/output"
    # Set the output S3 url for feature store query results
    output_query_location = f'{output_s3_prefix}/offline-store/query_results'
    
    # Set the output S3 urls for processed data
    train_s3_url = f"{output_s3_prefix}/train"
    validation_s3_url = f"{output_s3_prefix}/validation"
    test_s3_url = f"{output_s3_prefix}/test"
    evaluation_s3_url = f"{output_s3_prefix}/evaluation"
    
    baseline_s3_url = f"{output_s3_prefix}/baseline"
    prediction_baseline_s3_url = f"{output_s3_prefix}/prediction_baseline"
    
    # xgboost_image_uri = sagemaker.image_uris.retrieve(
    #         "xgboost", 
    #         region=region,
    #         version="1.5-1"
    # )

    # If no tracking server ARN, try to find an active MLflow server
    if tracking_server_arn is None:
        r = sm.list_mlflow_tracking_servers(
            TrackingServerStatus='Created',
        )['TrackingServerSummaries']
    
        if len(r) < 1:
            print("You don't have any running MLflow servers. Exiting...")
            return None
        else:
            tracking_server_arn = r[0]['TrackingServerArn']
            print(f"Use the tracking server ARN:{tracking_server_arn}")
        
    # Parameters for pipeline execution
    
    # Set processing instance type
    process_instance_type_param = ParameterString(
        name="ProcessingInstanceType",
        default_value=process_instance_type,
    )

    # Set training instance type
    train_instance_type_param = ParameterString(
        name="TrainingInstanceType",
        default_value=train_instance_type,
    )

    # Set model approval param
    model_approval_status_param = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"
    )

    # Minimal threshold for model performance on the test dataset
    test_score_threshold_param = ParameterFloat(
        name="TestScoreThreshold", 
        default_value=test_score_threshold
    )

    # S3 url for the input dataset
    input_s3_url_param = ParameterString(
        name="InputDataUrl",
        default_value=input_s3_url if input_s3_url else "None",
    )

    # Feature group name for the input featureset
    feature_group_name_param = ParameterString(
        name="FeatureGroupName",
        default_value=feature_group_name if feature_group_name else "None",
    )
    
    # Model package group name
    model_package_group_name_param = ParameterString(
        name="ModelPackageGroupName",
        default_value=model_package_group_name,
    )

    # MLflow tracking server ARN
    tracking_server_arn_param = ParameterString(
        name="TrackingServerARN",
        default_value=tracking_server_arn,
    )
    
    # Define step cache config
    cache_config = CacheConfig(
        enable_caching=True,
        expire_after="P30d" # 30-day
    )

    # Construct the pipeline
    
    # Get datasets
    step_get_datasets = step(
            preprocess, 
            role=role,
            instance_type=process_instance_type_param,
            name=f"preprocess",
            keep_alive_period_in_seconds=3600,
    )(
        input_data_s3_path=input_s3_url_param,
        output_s3_prefix=output_s3_prefix,
        tracking_server_arn=tracking_server_arn_param,
        experiment_name=experiment_name,
        pipeline_run_name=ExecutionVariables.PIPELINE_EXECUTION_ID,
    ) if input_s3_url else step(
        prepare_datasets, 
        role=role,
        instance_type=process_instance_type_param,
        name=f"extract-featureset",
        keep_alive_period_in_seconds=3600,
    )(
        feature_group_name=feature_group_name_param,
        output_s3_prefix=output_s3_prefix,
        query_output_s3_path=output_query_location,
        tracking_server_arn=tracking_server_arn_param,
        experiment_name=experiment_name,
        pipeline_run_name=ExecutionVariables.PIPELINE_EXECUTION_ID,
    )
    
    # Instantiate an XGBoost estimator object
    # estimator = sagemaker.estimator.Estimator(
    #     image_uri=xgboost_image_uri,
    #     role=role, 
    #     instance_type=train_instance_type_param,
    #     instance_count=1,
    #     output_path=output_s3_url,
    #     sagemaker_session=session,
    #     base_job_name=f"{pipeline_name}-train"
    # )
    
    # # Define algorithm hyperparameters
    # estimator.set_hyperparameters(
    #     num_round=100, # the number of rounds to run the training
    #     max_depth=3, # maximum depth of a tree
    #     eta=0.5, # step size shrinkage used in updates to prevent overfitting
    #     alpha=2.5, # L1 regularization term on weights
    #     objective="binary:logistic",
    #     eval_metric="auc", # evaluation metrics for validation data
    #     subsample=0.8, # subsample ratio of the training instance
    #     colsample_bytree=0.8, # subsample ratio of columns when constructing each tree
    #     min_child_weight=3, # minimum sum of instance weight (hessian) needed in a child
    #     early_stopping_rounds=10, # the model trains until the validation score stops improving
    #     verbosity=1, # verbosity of printing messages
    # )

    estimator = SKLearn(
        entry_point='train.py',
        source_dir='./training',
        framework_version="0.23-1",  
        py_version='py3',
        #hyperparameters=hyperparams,
        role=role,
        instance_count=1,
        instance_type=train_instance_type_param,
        output_path=output_s3_url,
        #base_job_name="from-idea-to-prod-training",
        sagemaker_session=session,
        base_job_name=f"{pipeline_name}-train"
        #environment=env_variables,
    )

    
    
    # train step
    step_train = TrainingStep(
        name=f"train",
        step_args=estimator.fit(
            {
                "train": TrainingInput(
                    step_get_datasets['train_data'],
                    content_type="text/csv",
                ),
                "validation": TrainingInput(
                    step_get_datasets['validation_data'],
                    content_type="text/csv",
                ),
            }
        ),
        cache_config=cache_config,
    )   
    
    # Evaluation step
    step_evaluate = step(
        evaluate,
        role=role,
        instance_type=process_instance_type_param,
        name=f"evaluate",
        keep_alive_period_in_seconds=3600,
    )(
        test_x_data_s3_path=step_get_datasets['test_x_data'],
        #test_y_data_s3_path=step_get_datasets['test_y_data'],
        model_s3_path=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        output_s3_prefix=output_s3_prefix,
        tracking_server_arn=tracking_server_arn_param,
        experiment_name=step_get_datasets['experiment_name'],
        pipeline_run_id=step_get_datasets['pipeline_run_id'],
    )

    # register model step
    step_register = step(
        register,
        role=role,
        instance_type=process_instance_type_param,
        name=f"register",
        keep_alive_period_in_seconds=3600,
    )(
        training_job_name=step_train.properties.TrainingJobName,
        model_package_group_name=model_package_group_name_param,
        model_approval_status=model_approval_status_param,
        evaluation_result=step_evaluate['evaluation_result'],
        output_s3_prefix=output_s3_url,
        tracking_server_arn=tracking_server_arn_param,
        experiment_name=step_get_datasets['experiment_name'],
        pipeline_run_id=step_get_datasets['pipeline_run_id'],
    )

    # fail the pipeline execution step
    step_fail = FailStep(
        name=f"fail",
        error_message=Join(on=" ", values=["Execution failed due to AUC Score < ", test_score_threshold_param]),
    )
    
    # condition to check in the condition step
    condition_gte = ConditionGreaterThanOrEqualTo(
            left=step_evaluate['evaluation_result']['classification_metrics']['auc_score']['value'],  
            right=test_score_threshold_param,
    )
    
    # conditional register step
    step_conditional_register = ConditionStep(
        name=f"check-metrics",
        conditions=[condition_gte],
        if_steps=[step_register],
        else_steps=[step_fail],
    )   

    # Create a pipeline object
    pipeline = Pipeline(
        name=f"{pipeline_name}",
        parameters=[
            input_s3_url_param,
            feature_group_name_param,
            process_instance_type_param,
            train_instance_type_param,
            model_approval_status_param,
            test_score_threshold_param,
            model_package_group_name_param,
            tracking_server_arn_param,
        ],
        steps=[step_conditional_register],
        pipeline_definition_config=PipelineDefinitionConfig(use_custom_job_prefix=True)
    )
    
    return pipeline
