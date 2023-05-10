from model_class import simple_model
from keras.metrics import Precision, Recall, F1Score


def train_model():
    print("Setting up Parameters")
    batch_size = 64
    epochs = 120
    
    # load train data
    train_data = []
    validation_data = []

    print("Training model...")
    model = simple_model()
    model.compile(loss='binary_crossentropy',
                  optimizer = "Adam",
                  metrics=['accuracy',Precision(),Recall(),F1Score()])
    
    history = model.fit(
        train_data,
        epochs = epochs,
        batch_size = batch_size,
        validation_data=validation_data,
        verbose = 1
    )

    model.save('./data/model.h5')
