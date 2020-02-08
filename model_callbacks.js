async function trainModel(xTrain, yTrain, xTest, yTest){
    const model = tf.sequential();
    const learningRate = .01;
    const numberOfEpochs = 40;
    const optimizer = tf.train.adam(learningRate);

    model.add(tf.layers.dense(
        {units: 10, activation: 'sigmoid', inputShape: [xTrain.shape[1]]}
    ));

    model.add(tf.layers.dense(
        {units: 3, activation: 'softmax'}
    ));

    model.compile(
        {optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy']}
    );

    const history = await model.fit(xTrain, yTrain, 
        { epochs: numberOfEpochs, validationData: [xTest, yTest],
            callbacks: {
                onEpochEnd: async (epoch, logs) => {
                    console.log(`Epoch: ${epoch} Logs: ${logs.loss}`);
                    await tf.nextFrame();
                }
            }
        });
    
    return model;
}

async function doIris(){
    const [xTrain, yTrain, xTest, yTest] = getIrisData(.2);

    model = await trainModel(xTrain, yTrain, xTest, yTest);

    const input = tf.tensor2d([6.8, 2.8, 4.8, 1.4], [1, 4]);
    const prediction = model.predict(input);
    alert(prediction);

    const predictionWithArgMax = model.predict(input).argMax(-1).dataSync();
    alert(predictionWithArgMax);

    const xData = xTest.dataSync();
    const yTrue = yTest.argMax(-1).dataSync();

    const predictions = await model.predict(xTest);
    const yPred = predictions.argMax(-1).dataSync();

    let correct = 0;
    let wrong = 0;

    for(let i = 0; i < yTrue.length; i++){
        if(yTrue[i] == yPred[i]){
            correct++;
        } else {
            wrong++;
        }
    }

    alert(`Correct Preds: ${correct} | Wrong Preds: ${wrong} | Prediction Rate: ${(wrong / yTrue.length)}`);
}

doIris();