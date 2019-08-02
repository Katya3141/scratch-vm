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

const classifier = knnClassifierModule.create();


/**
 * Class for the teachable classifier blocks.
 * @constructor
 */
class Scratch3TeachableClassifierNN {
    constructor (runtime) {
        this.runtime = runtime;
        this.predictedLabel = null;
        this.labelList = [''];
        this.labelListEmpty = true;
        this.oldLabelList = [];
        this.mobilenetModule = null;
        this.model = null;
        this.xs = null;
        this.ys = null;

        tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json').then(net => {
            const layer = net.getLayer('conv_pw_13_relu');
            this.mobilenetModule = tf.model({inputs: net.inputs, outputs: layer.output});

            if (this.runtime.ioDevices) {
                // Kick off looping the analysis logic.
                this._loop();
            }
        });

        /**
         * The last millisecond epoch timestamp that the video stream was
         * analyzed.
         * @type {number}
         */
        this._lastUpdate = null;
    }

    /**
     * After analyzing a frame the amount of milliseconds until another frame
     * is analyzed.
     * @type {number}
     */
    static get INTERVAL () {
        return 100;
    }

    /**
     * Dimensions the video stream is analyzed at after its rendered to the
     * sample canvas.
     * @type {Array.<number>}
     */
    static get DIMENSIONS () {
        return [224, 224];
    }

    /**
     * Occasionally step a loop to sample the video and predict the current label
     * @private
     */
    _loop () {
        setTimeout(this._loop.bind(this), Math.max(this.runtime.currentStepTime, Scratch3TeachableClassifierNN.INTERVAL));

        const time = Date.now();
        if (this._lastUpdate === null) {
            this._lastUpdate = time;
        }
        const offset = time - this._lastUpdate;
        if (offset > Scratch3TeachableClassifierNN.INTERVAL) {
            if (this.model && this.oldLabelList.length > 0) {
                const frame = this.runtime.ioDevices.video.getFrame({
                    format: Video.FORMAT_IMAGE_DATA,
                    dimensions: Scratch3TeachableClassifierNN.DIMENSIONS
                });
                if (frame) {
                    const result = this.model.predict(this.mobilenetModule.predict(tf.browser.fromPixels(frame).expandDims(0)));   //predict
                    const prediction = result.as1D().argMax();
                    prediction.data().then(data => {
                        this.predictedLabel = this.oldLabelList[data[0]];
                    })
                }
            } else {
                this.predictedLabel = '';
            }
        }
    }

    
    /**
     * @returns {object} metadata for this extension and its blocks.
     */
    getInfo () {
        this.runtime.ioDevices.video.enableVideo();
        this.runtime.ioDevices.video.setPreviewGhost(50);

        return {
            id: 'teachableClassifierNN',
            name: 'Teachable Classifier NN',
            blocks: [
                {
                    opcode: 'whenISee',
                    text: 'when I see [LABEL]',
                    blockType: BlockType.HAT,
                    arguments: {
                        LABEL: {
                            type: ArgumentType.STRING,
                            menu: 'LABEL',
                            defaultValue: ''
                        }
                    }
                },
                {
                    opcode: 'imageExample',
                    text: 'add example with label [LABEL]',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        LABEL: {
                            type: ArgumentType.STRING,
                            defaultValue: 'background'
                        }
                    }
                },
                {
                    opcode: 'train',
                    text: 'train with these examples',
                    blockType: BlockType.COMMAND,
                },
                {
                    opcode: 'predictImageLabel',
                    text: 'predict label',
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
                            menu: 'LABEL',
                            defaultValue: ''
                        }
                    }
                },
                {
                    opcode: 'controlVideo',
                    text: 'turn video [STATE]',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        STATE: {
                            type: ArgumentType.STRING,
                            menu: 'VIDEO',
                            defaultValue: 'on'
                        }
                    }
                }
            ],
            menus: {
                LABEL: 'getLabels',
                VIDEO: ['on', 'off', 'on flipped']
            }
        };
    }

    getLabels () {
        return this.labelList;
    }

    toOneHot (labels) {
        let oneHotLabels = [];
        for(let i = 0; i < labels.length; i++) {
            let l = Array(this.oldLabelList.length).fill(0);
            l[this.oldLabelList.indexOf(labels[i])] = 1;
            oneHotLabels.push(l);
        }
        return oneHotLabels;
    }

    whenISee (args) {
        return this.predictedLabel === args.LABEL;
    }

    train () {
        if (this.xs) {
            for (let i = 0; i < this.labelList.length; i++) {
                this.oldLabelList.push(this.labelList[i]);
            }
            const numClasses = this.oldLabelList.length;

            const xsTensor = this.xs
            const ysTensor = tf.tensor(this.toOneHot(this.ys));

            this.model = tf.sequential();
            this.model.add(tf.layers.flatten({inputShape: this.mobilenetModule.outputs[0].shape.slice(1)}));
            this.model.add(tf.layers.dense({units:100, activation:'relu'}));
            this.model.add(tf.layers.dense({units:numClasses, activation:'softmax'}));

            const learningRate = 0.0001;
            const optimizer = tf.train.adam(learningRate);
            this.model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

            this.model.fit(xsTensor, ysTensor, {
                batchSize: 10,
                epochs: 20
            });
        } else {
            this.oldLabelList = [];
        }
    }

    imageExample (args) {
        if (this.mobilenetModule !== null) {
            const frame = this.runtime.ioDevices.video.getFrame({
                format: Video.FORMAT_IMAGE_DATA,
                dimensions: Scratch3TeachableClassifierNN.DIMENSIONS
            });
            if (frame) {
                const example = this.mobilenetModule.predict(tf.browser.fromPixels(frame).expandDims(0)); //add example
                if (this.xs) {
                    this.xs = this.xs.concat(example);
                    this.ys = this.ys.concat(args.LABEL);
                } else {
                    this.xs = example;
                    this.ys = [args.LABEL];
                }
                if (this.labelListEmpty) {
                    this.labelList.splice(this.labelList.indexOf(''), 1);
                    this.labelListEmpty = false;
                }
                if (!this.labelList.includes(args.LABEL)) {
                    this.labelList.push(args.LABEL);
                }
            }
        }
    }

    predictImageLabel () {
        return this.predictedLabel;
    }

    clearAll () {
        this.xs = null;
        this.ys = null;
        this.labelList = [''];
        this.labelListEmpty = true;
    }

    clearAllWithLabel (args) {
        for (let i = this.xs.length - 1; i >= 0; i--) {
            if (this.ys[i] === args.LABEL) {
                this.xs.splice(i, 1);
                this.ys.splice(i, 1);
            }
        }
        this.labelList.splice(this.labelList.indexOf(args.LABEL), 1);
        if (this.labelList.length === 0) {
            this.labelListEmpty = true;
            this.labelList.push('');
        }
    }

    controlVideo (args) {
        if (args.STATE === 'off') {
            this.runtime.ioDevices.video.disableVideo();
        } else {
            this.runtime.ioDevices.video.enableVideo();
            this.runtime.ioDevices.video.mirror = args.STATE === 'on';
        }
    }
}
module.exports = Scratch3TeachableClassifierNN;
