const formatMessage = require('format-message');
const nets = require('nets');

const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const Cast = require('../../util/cast');
const MathUtil = require('../../util/math-util');
const Clone = require('../../util/clone');
const Video = require('../../io/video');
const log = require('../../util/log');
const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const knnClassifierModule = require('@tensorflow-models/knn-classifier');
//const w2v = require('word2vec');

const classifier = knnClassifierModule.create();
let mobilenetLoaded = false;
let mobilenetModule = null;
mobilenet.load().then(m => {
    mobilenetLoaded = true;
    mobilenetModule = m;
});

let uniqueChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 .,;:!?'\"-()/";
let vocabSize = uniqueChars.length;

/**
 * Class for the teachable classifier blocks.
 * @constructor
 */
class Scratch3TeachableClassifier {
    constructor (runtime) {
        this.runtime = runtime;
    }

    
    /**
     * @returns {object} metadata for this extension and its blocks.
     */
    getInfo () {
        this.runtime.ioDevices.video.enableVideo();

        return {
            id: 'teachableClassifier',
            name: 'Teachable Classifier',
            blocks: [
                {
                    opcode: 'imageExample',
                    text: 'add example from video with label [LABEL]',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        LABEL: {
                            type: ArgumentType.STRING,
                            defaultValue: 'class 1'
                        }
                    }
                },
                {
                    opcode: 'predictImageLabel',
                    text: 'predict image label',
                    blockType: BlockType.REPORTER
                },
                {
                    opcode: 'clearAll',
                    text: 'clear all examples',
                    blockType: BlockType.COMMAND
                },
                {
                    opcode: 'clearAllWithLabel',
                    text: 'clear all examples with label [LABEL]',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        LABEL: {
                            type: ArgumentType.STRING,
                            defaultValue: 'class 1'
                        }
                    }
                }
            ],
        };
    }

    toOneHot (inputText) { //Converts a string to a one-hot tensor to be used as input to the classifier
        let tempArray = [];
        let oneHotArray = []
        for (const c of inputText) {
            tempArray.push(uniqueChars.indexOf(c));
        }
        for (let i = 0; i < tempArray.length; i++) {
            oneHotArray[i] = new Array(vocabSize).fill(0);
            oneHotArray[i][tempArray[i]] = 1;
        }
        return tf.tensor([oneHotArray]);
    }

    imageExample (args) {
        if (mobilenetLoaded) {  //check that the mobilenet has loaded
            const frame = this.runtime.ioDevices.video.getFrame({
                format: Video.FORMAT_IMAGE_DATA,
                dimensions: [480, 360]
            });
            const example = mobilenetModule.infer(frame); //add example
            classifier.addExample(example, args.LABEL);
        } else {
            return '[still loading module...]'   //if mobilenet not loaded yet, return "still loading" message
        }
    }

    predictImageLabel () {
        if (classifier.getNumClasses() > 0) {
            const wordPromise = new Promise(resolve => {
                if (mobilenetLoaded) {  //check that the mobilenet has loaded
                    const frame = this.runtime.ioDevices.video.getFrame({
                        format: Video.FORMAT_IMAGE_DATA,
                        dimensions: [480, 360]
                    });
                    input = mobilenetModule.infer(frame);   //predict
                    classifier.predictClass(input).then(result => {
                        resolve(result.label);
                        return result.label;
                    })
                } else {
                    resolve('[still loading module...]');    //if mobilenet not loaded yet, return "still loading" message
                    return '[still loading module...]';
                }
            });
            return wordPromise;
        }
        return '[add examples to help me predict!]' //if there aren't any examples yet, return "add examples" message
    }

    clearAll () {
        classifier.clearAllClasses();   //clear all examples
    }

    clearAllWithLabel (args) {
        if (classifier.getClassExampleCount(args.LABEL) > 0) {
            classifier.clearClass(args.LABEL);  //clear examples with a certain label
        }
    }
}
module.exports = Scratch3TeachableClassifier;
