import mlflow
import mlflow.mleap
import mlflow.sagemaker as mfs
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer


# Prepare training documents from a list of (id, text, label) tuples.
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])


# Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])


# Fit the pipeline to training documents.
model = pipeline.fit(training)


# Prepare test documents, which are unlabeled (id, text) tuples.
test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])


# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))


# Start a new MLflow run
with mlflow.start_run() as run:
    # Log the model within the MLflow run
    mlflow.mleap.log_model(spark_model=model, sample_input=test, artifact_path="model")


# Set AWS execution role and ECR image url
app_name = "spark-mleap"
region = "us-east-1"
model_uri = "runs:/" + "50fc9f4517e74c549174aee994b18713" + "/model"
arn = "arn:aws:iam::197306934454:role/service-role/AmazonSageMaker-ExecutionRole-20200429T114488"
image_ecr_url = "197306934454.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:1.8.0"


mfs.deploy(app_name=app_name, 
	model_uri=model_uri, 
	image_url=image_ecr_url, 
	mode="replace", 
	flavor="mleap", 
	region_name=region, 
	execution_role_arn = arn)