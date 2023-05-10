"""
Evaluate model performace
"""
import json
from keras.models import load_model


def eval_model():

    print("Loading model and data....")
    #Load Test Data
    test_data = []

    #Loading model
    model = load_model("./data/model.h5")

    print("done.")

    print("Running model on test data...")
    results = model.evaluate(test_data, batch_size=64)
    print("done.")

    print("Calculating metrics....")
    metrics = dict(zip(model.metric_names,results))

    json_object = json.dumps(metrics, indent = 4)
    with open('./metrics/eval.json','w') as f:
        f.write(json_object)

    print("done.")

if __name__=='__main__':
    eval_model()

