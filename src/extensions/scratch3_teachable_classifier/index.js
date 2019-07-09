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
class Scratch3TeachableClassifier {
    constructor (runtime) {
        this.runtime = runtime;
        this.predictedLabel = null;
        this.labelList = [' '];
        this.mobilenetModule = null;

        mobilenet.load(1, 0.5).then(net => {
            this.mobilenetModule = net;
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
        return [480, 360];
    }

    /**
     * Occasionally step a loop to sample the video and predict the current label
     * @private
     */
    _loop () {
        setTimeout(this._loop.bind(this), Math.max(this.runtime.currentStepTime, Scratch3TeachableClassifier.INTERVAL));

        const time = Date.now();
        if (this._lastUpdate === null) {
            this._lastUpdate = time;
        }
        const offset = time - this._lastUpdate;
        if (offset > Scratch3TeachableClassifier.INTERVAL) {
            if (classifier.getNumClasses() > 0) {
                const frame = this.runtime.ioDevices.video.getFrame({
                    format: Video.FORMAT_IMAGE_DATA,
                    dimensions: Scratch3TeachableClassifier.DIMENSIONS
                });
                if (frame) {
                    input = this.mobilenetModule.infer(frame);   //predict
                    classifier.predictClass(input).then(result => {
                        this.predictedLabel = result.label;
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

        return {
            id: 'teachableClassifier',
            name: 'Teachable Classifier',
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
            menus: {
                LABEL: 'getLabels'
            }
        };
    }

    getLabels () {
        return this.labelList;
    }

    whenISee (args) {
        return this.predictedLabel == args.LABEL;
    }

    imageExample (args) {
        if (this.mobilenetModule != null) {
            const frame = this.runtime.ioDevices.video.getFrame({
                format: Video.FORMAT_IMAGE_DATA,
                dimensions: Scratch3TeachableClassifier.DIMENSIONS
            });
            const example = this.mobilenetModule.infer(frame); //add example
            classifier.addExample(example, args.LABEL);
            if (!this.labelList.includes(args.LABEL)) {
                this.labelList.push(args.LABEL);
            }
        }
    }

    predictImageLabel () {
        return this.predictedLabel;
    }

    clearAll () {
        classifier.clearAllClasses();   //clear all examples
        this.labelList = [' '];
    }

    clearAllWithLabel (args) {
        if (classifier.getClassExampleCount()[args.LABEL] > 0) {
            classifier.clearClass(args.LABEL);  //clear examples with a certain label
            this.labelList.splice(this.labelList.indexOf(args.LABEL), 1);
        }
    }
}
module.exports = Scratch3TeachableClassifier;
