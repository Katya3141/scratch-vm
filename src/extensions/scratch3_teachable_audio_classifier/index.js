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
const speechCommands = require('@tensorflow-models/speech-commands');

/**
 * Class for the teachable classifier blocks.
 * @constructor
 */
class Scratch3TeachableAudioClassifierBlocks {
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

    constructor (runtime) {
        this.runtime = runtime;
        this.listening = false;
        this.predictedLabel = null;
        this.labelList = [''];
        this.labelListEmpty = true;
        this.recognizer = speechCommands.create('BROWSER_FFT');
        this.transferRecognizer = null;
        this.recognizer.ensureModelLoaded().then(model => {
            this.transferRecognizer = this.recognizer.createTransfer('words');
        });

        if (this.runtime.ioDevices) {
            // Kick off looping the analysis logic.
            this._loop();
        }

        /**
         * The last millisecond epoch timestamp that the video stream was
         * analyzed.
         * @type {number}
         */
        this._lastUpdate = null;
    }

    /**
     * Occasionally step a loop to sample the video and predict the current label
     * @private
     */
    _loop () {
        setTimeout(this._loop.bind(this), Math.max(this.runtime.currentStepTime, Scratch3TeachableAudioClassifierBlocks.INTERVAL));

        const time = Date.now();
        if (this._lastUpdate === null) {
            this._lastUpdate = time;
        }
        const offset = time - this._lastUpdate;
        if (offset > Scratch3TeachableAudioClassifierBlocks.INTERVAL) {
            if (!this.labelListEmpty) {   //whenever the classifier has some data
                //TODO listen/recognize and classify the label
                // if (!this.listening) {
                //     this.runtime.emitMicListening(true);
                //     this.listening = true;
                //     this.transferRecognizer.ensureModelLoaded().then(resolve => {    //when the model is loaded, start listening, and continuously choose the most likely command
                //         this.transferRecognizer.listen(result => {
                //             let maxScore = Math.max(...result.scores);
                //             let command = this.transferRecognizer.wordLabels()[result.scores.indexOf(maxScore)];
                //             if (maxScore > 0.8 && command !== '_background_noise_' && command !== '_unknown_') {  //if it recognized a word with high enough probability, set both commands to that word; otherwise reset newCommand
                //                 console.log(command);
                //                 this.predictedLabel = command;
                //             } else {
                //                 this.predictedLabel = '';
                //             }
                //             this.listening = false;
                //         }, {
                //             probabilityThreshold: 0,
                //             invokeCallbackOnNoiseAndUnknown: true
                //         });
                //     });
                // }
            } else {
                this.predictedLabel = '';
            }
        }
    }

    
    /**
     * @returns {object} metadata for this extension and its blocks.
     */
    getInfo () {
        return {
            id: 'teachableAudioClassifier',
            name: 'Teachable Audio Classifier',
            blocks: [
                {
                    func: 'EDIT_MODEL',
                    blockType: BlockType.BUTTON,
                    text: 'Edit Model'
                },
                {
                    opcode: 'whenIHear',
                    text: 'when [LABEL] heard',
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
                    opcode: 'predictAudioLabel',
                    text: 'predict label',
                    blockType: BlockType.REPORTER
                },
                {
                    opcode: 'addExample',
                    text: 'add example with label [LABEL]',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        LABEL: {
                            type: ArgumentType.STRING,
                            defaultValue: ''
                        }
                    }
                },
                {
                    opcode: 'train',
                    text: 'train',
                    blockType: BlockType.COMMAND
                }
            ],
            menus: {
                LABEL: 'getLabels'
            }
        };
    }

    getLabels () {  //return label list for block menus
        return this.labelList;
    }

    whenIHear (args) {   //check if the current predicted label matches the query
        return this.predictedLabel === args.LABEL;
    }

    addExample (args) {
        //TODO listen/recognize and add example
        this.transferRecognizer.collectExample(args.LABEL.toString());
        console.log(this.transferRecognizer);


        if (this.labelListEmpty) {
            this.labelList.splice(this.labelList.indexOf(''), 1);   //edit label list accordingly
            this.labelListEmpty = false;
        }
        if (!this.labelList.includes(args.LABEL)) {
            this.labelList.push(args.LABEL);
        }
    }

    predictAudioLabel () {  //return current predicted label
        return this.predictedLabel;
    }

    train () {
        //TODO train with current examples
        console.log(this.transferRecognizer.countExamples());
        this.transferRecognizer.train({
            epochs: 25
        }).then(() => {
            console.log("trained!");
            this.listening = true;
        })
    }
}
module.exports = Scratch3TeachableAudioClassifierBlocks;
