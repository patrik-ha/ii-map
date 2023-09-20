function getMaxFromChannel(prediction, i, j) {
    let vals = [];
    for (let p = 0; p < 12; p++) {
        vals.push(prediction[p][i][j]);
    }
    return Math.max.apply(Math, vals);
}


function plane(fillValue) {
    let sample = [];
    for (let i = 0; i < 8; i++) {
        sample.push([]);
        for (let j = 0; j < 8; j++) {
            sample[i].push(fillValue);
        }
    }
    return sample;
}

function emptySample() {
    let sample = [];
    for (let p = 0; p < 20; p++) {
        sample.push(plane(0));
    }
    return sample;
    // return tf.tensor([sample], [1, 20, 8, 8]);
}