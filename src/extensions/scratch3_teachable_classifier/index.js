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

/**
 * Class for the teachable classifier blocks.
 * @constructor
 */
class Scratch3TeachableClassifierBlocks {
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
        this.runtime.modelData.nextLabelNumber = 1;
        this.predictedLabel = null;
        this.labelList = [];
        this.labelListEmpty = true;
        this.mobilenetModule = null;
        this.classifier = knnClassifierModule.create();

        this.loadModelFromRuntime();

        mobilenet.load(2, 0.5).then(net => {
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

        this.runtime.on('PROJECT_LOADED', () => {
            this.clearLocal();
            this.loadModelFromRuntime();
        })

        //listen for events emitted from the GUI
        this.runtime.on('DELETE_EXAMPLE', (label, exampleNum) => {
            this.deleteExample(label, exampleNum);
        });
        this.runtime.on('DELETE_LOADED_EXAMPLES', (label) => {
            this.deleteLoadedExamples(label);
        });
        this.runtime.on('DELETE_LABEL', (label) => {
            this.clearAllWithLabel({LABEL: label});
        });
        this.runtime.on('NEW_EXAMPLES', (examples, label) => {
            this.newExamples(examples, label);
        });
        this.runtime.on('CLEAR_ALL_LABELS', () => {
            this.clearAll();
        });
        this.runtime.on('RENAME_LABEL', (oldName, newName) => {
            this.renameLabel(oldName, newName);
        });
    }

    /**
     * Occasionally step a loop to sample the video and predict the current label
     * @private
     */
    _loop () {
        setTimeout(this._loop.bind(this), Math.max(this.runtime.currentStepTime, Scratch3TeachableClassifierBlocks.INTERVAL));

        const time = Date.now();
        if (this._lastUpdate === null) {
            this._lastUpdate = time;
        }
        const offset = time - this._lastUpdate;
        if (offset > Scratch3TeachableClassifierBlocks.INTERVAL) {
            if (this.classifier.getNumClasses() > 0) {   //whenever the classifier has some data
                const frame = this.runtime.ioDevices.video.getFrame({
                    format: Video.FORMAT_IMAGE_DATA,
                    dimensions: Scratch3TeachableClassifierBlocks.DIMENSIONS
                });
                if (frame) {
                    const input = this.mobilenetModule.infer(frame);   //predict
                    this.classifier.predictClass(input).then(result => {
                        this.predictedLabel = result.label;
                    });
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
            id: 'teachableClassifier',
            name: 'Teachable Classifier',
            blocks: [
                {
                    func: 'EDIT_MODEL',
                    blockType: BlockType.BUTTON,
                    text: 'Edit Model'
                },
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
                    opcode: 'predictImageLabel',
                    text: 'predict label',
                    blockType: BlockType.REPORTER
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
                },
                '---',
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
                    opcode: 'clearAll',
                    text: 'clear all examples',
                    blockType: BlockType.COMMAND
                }
            ],
            menus: {
                LABEL: 'getLabels',
                VIDEO: ['on', 'off', 'on flipped']
            }
        };
    }

    loadModelFromRuntime () {
        this.labelList = [];
        let classifierData = {...this.runtime.modelData.classifierData};
        for (let label of Object.keys(classifierData)) {
            classifierData[label] = tf.tensor(classifierData[label]);
            this.labelList.push(label);
            this.labelListEmpty = false;
        }
        if (this.labelListEmpty) {
            this.labelList.push('');
        }
        this.classifier.clearAllClasses();
        this.classifier.setClassifierDataset({...classifierData});
    }

    getLabels () {  //return label list for block menus
        return this.labelList;
    }

    whenISee (args) {   //check if the current predicted label matches the query
        return this.predictedLabel === args.LABEL;
    }

    imageExample (args) {   //take a picture and add it as an example
        if (this.mobilenetModule !== null) {
            const frame = this.runtime.ioDevices.video.getFrame({
                format: Video.FORMAT_IMAGE_DATA,
                dimensions: Scratch3TeachableClassifierBlocks.DIMENSIONS
            });
            if (frame) {
                this.newExamples([frame], args.LABEL);
            }
        }
    }

    newExamples (images, label) {
        for (let image of images) {
            const example = this.mobilenetModule.infer(image);
            const exampleArray = tf.div(example, example.norm()).arraySync()[0];
            this.classifier.addExample(example, label);  //add example to the classifier
            if (this.labelListEmpty) {
                this.labelList.splice(this.labelList.indexOf(''), 1);   //edit label list accordingly
                this.labelListEmpty = false;
            }
            if (!this.labelList.includes(label)) {
                this.labelList.push(label);
                this.runtime.modelData.imageData[label] = [image];    //update the runtime's model data (to share with the GUI)
                this.runtime.modelData.classifierData[label] = [exampleArray];
            } else {
                this.runtime.modelData.imageData[label].push(image);
                this.runtime.modelData.classifierData[label].push(exampleArray);
            }
        }
    }

    renameLabel (oldName, newName) {
        let data = {...this.classifier.getClassifierDataset()};  //reset the classifier dataset with the renamed label
        if (data[oldName]) {
            data[newName] = data[oldName];
            delete data[oldName];
            this.classifier.clearAllClasses();
            this.classifier.setClassifierDataset(data);
        }

        this.runtime.modelData.classifierData[newName] = this.runtime.modelData.classifierData[oldName];  //reset the runtime's model data with the new renamed label (to share with GUI)
        delete this.runtime.modelData.classifierData[oldName];

        this.runtime.modelData.imageData[newName] = this.runtime.modelData.imageData[oldName];  //reset the runtime's model data with the new renamed label (to share with GUI)
        delete this.runtime.modelData.imageData[oldName];

        this.labelList.splice(this.labelList.indexOf(oldName), 1);  //reset label list with the new renamed label
        this.labelList.push(newName)
    }

    deleteExample (label, exampleNum) {
        this.runtime.modelData.imageData[label].splice(exampleNum, 1);
        this.runtime.modelData.classifierData[label].splice(exampleNum - this.runtime.modelData.imageData[label].length - 1, 1);    //imageData[label].length is ONLY the length of the NEW examples (not the saved and then loaded ones!)

        let data = {...this.classifier.getClassifierDataset()};  //reset the classifier dataset with the deleted example
        let labelExamples = data[label].arraySync();
        labelExamples.splice(exampleNum - this.runtime.modelData.imageData[label].length - 1, 1);   //imageData[label].length is ONLY the length of the NEW examples (not the saved and then loaded ones!)
        if (labelExamples.length > 0) {
            data[label] = tf.tensor(labelExamples);
            this.classifier.clearAllClasses();
            this.classifier.setClassifierDataset(data);
        } else {
            this.classifier.clearClass(label);
        }
    }

    deleteLoadedExamples (label) {
        let numLoadedExamples = this.runtime.modelData.classifierData[label].length - this.runtime.modelData.imageData[label].length;   //imageData[label].length is ONLY the length of the NEW examples (not the saved and then loaded ones!)
        this.runtime.modelData.classifierData[label].splice(0, numLoadedExamples);

        let data = {...this.classifier.getClassifierDataset()};  //reset the classifier dataset with the deleted examples
        let labelExamples = data[label].arraySync();
        labelExamples.splice(0, numLoadedExamples);
        if (labelExamples.length > 0) {
            data[label] = tf.tensor(labelExamples);
            this.classifier.clearAllClasses();
            this.classifier.setClassifierDataset(data);
        } else {
            this.classifier.clearClass(label);
        }
    }

    predictImageLabel () {  //return current predicted label
        return this.predictedLabel;
    }

    clearLocal () {
        this.classifier.clearAllClasses();
        this.labelList = [''];
        this.labelListEmpty = true;
        this.runtime.modelData.nextLabelNumber = 1;
    }

    clearAll () {
        this.clearLocal();
        this.runtime.modelData = {imageData: {}, classifierData: {}, nextLabelNumber: 1};    //clear runtime's model data
    }

    clearAllWithLabel (args) {
        if (this.labelList.includes(args.LABEL)) {
            if (this.classifier.getClassExampleCount()[args.LABEL] > 0) {
                this.classifier.clearClass(args.LABEL);  //clear examples with a certain label in the classifier
            }
            this.labelList.splice(this.labelList.indexOf(args.LABEL), 1);   //remove label from labelList
            delete this.runtime.modelData.classifierData[args.LABEL];  //remove label in the runtime's model data (to share with the GUI)
            delete this.runtime.modelData.imageData[args.LABEL];
            if (this.labelList.length === 0) {  //if the label list is now empty, fill it with an empty string
                this.labelListEmpty = true;
                this.labelList.push('');
            }
        }
    }

    controlVideo (args) {   //make video on/off/mirrored
        if (args.STATE === 'off') {
            this.runtime.ioDevices.video.disableVideo();
        } else {
            this.runtime.ioDevices.video.enableVideo();
            this.runtime.ioDevices.video.mirror = args.STATE === 'on';
        }
    }
}
module.exports = Scratch3TeachableClassifierBlocks;
