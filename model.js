//Tensor Converter
function convertToTensors(data, targets, testSplit){
    const numExamples = data.length;
    if(numExamples !== targets.length){
        throw new Error();
    }

    const numTestExamples = Math.round(numExamples * testSplit);
    const numTrainExamples = numExamples - numTestExamples;

    const xDims = data[0].length;

    //Create a 2D `tf.tensor` to hold the feature data.
    const xs = tf.tensor2d(data, [numExamples, xDims]);

    //Create a 1D `tf.Tensor` to hold the labels and convert the number label
    //from the set {0, 1, 2} into one-hot encoding (.e.g.. 0 --> [1, 0, 0]).
    const ys = tf.oneHot(tf.tensor1d(targets).toInt(), IRIS_NUM_CLASSES);

    //Split the data into training and test sets
    const xTrain = xs.slice([0, 0], [numTrainExamples, xDims]);
    const xTest = xs.slice([numTrainExamples, 0], [numTestExamples, xDims]);
    const yTrain = ys.slice([0, 0], [numTrainExamples, IRIS_NUM_CLASSES]);
    const yTest = ys.slice([0, 0], [numTestExamples, IRIS_NUM_CLASSES]);

    return [xTrain, yTrain, xTest, yTest];
}

function getIrisData(testSplit){
    return tf.tidy(() => {
        const dataByClass = [];
        const targetsByClass = [];
        for(let i = 0; i < IRIS_CLASSES.length; i++){
            dataByClass.push([]);
            targetsByClass.push([]);
        }
        for(const example of IRIS_DATA){
            const target = example[example.length - 1];
            const data = example.slice(0, example.length - 1);
            dataByClass[target].push(data);
            targetsByClass[target].push(target);
        }

        console.log(dataByClass);
        console.log(targetsByClass);

        const xTrains = [];
        const yTrains = [];
        const xTests = [];
        const yTests = [];
        for(let i = 0; i < IRIS_CLASSES.length; i++){
            const [xTrain, yTrain, xTest, yTest] = 
                convertToTensors(dataByClass[i], targetsByClass[i], testSplit);
            xTrains.push(xTrain);
            yTrains.push(yTrain);
            xTests.push(xTest);
            yTests.push(yTest);
        }

        const concatAxis = 0;
        return [
            tf.concat(xTrains, concatAxis), tf.concat(yTrains, concatAxis),
            tf.concat(xTests, concatAxis), tf.concat(yTests, concatAxis)
        ];
    });
}
