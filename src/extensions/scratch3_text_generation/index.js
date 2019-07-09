const formatMessage = require('format-message');
const nets = require('nets');

const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const Cast = require('../../util/cast');
const MathUtil = require('../../util/math-util');
const Clone = require('../../util/clone');
const log = require('../../util/log');
const tf = require('@tensorflow/tfjs');

let uniqueChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 \n.,;:!?'\"-()/";
let vocabSize = uniqueChars.length;


/**
 * Class for the text generation blocks.
 * @constructor
 */
class Scratch3TextGeneration {
    constructor (runtime) {
        this.craziness = 10;
        this.mostRecentChars = '';
        this.source = 'Dr. Seuss';

        this.MAX_GEN_LEN = 200;
    }

    
    /**
     * @returns {object} metadata for this extension and its blocks.
     */
    getInfo () {
        return {
            id: 'textGeneration',
            name: 'Text Generation',
            
            blocks: [
                {
                    opcode: 'genWords',
                    text: '[LENGTH] words starting with [SEED]',
                    blockType: BlockType.REPORTER,
                    arguments: {
                        LENGTH: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 10
                        },
                        SEED: {
                            type: ArgumentType.STRING,
                            defaultValue: 'Scratch '
                        }
                    }
                },
                {
                    opcode: 'genNextWord',
                    text: 'next word',
                    blockType: BlockType.REPORTER,
                    disableMonitor: true
                },
                {
                    opcode: 'genNextSentence',
                    text: 'next sentence',
                    blockType: BlockType.REPORTER,
                    disableMonitor: true
                },
                {
                    opcode: 'setCraziness',
                    text: 'set craziness to [CRAZINESS]%',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        CRAZINESS: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 10
                        }
                    }
                },
                {
                    opcode: 'setSource',
                    text: 'set source to [SOURCE]',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        SOURCE: {
                            type: ArgumentType.STRING,
                            menu: 'sourceText',
                            defaultValue: 'Dr. Seuss'
                        }
                    }
                },
                {
                    opcode: 'getCraziness',
                    text: 'craziness',
                    blockType: BlockType.REPORTER
                },
                {
                    opcode: 'getSource',
                    text: 'source',
                    blockType: BlockType.REPORTER
                }
            ],
            menus: {
                sourceText: ['Dr. Seuss', 'Shakespeare', 'jokes', 'Warriors', 'Moby Dick']
            }
        };
    }

    toOneHot(inputText) { //Converts a string to a one-hot array to be used in training the model
        let tempArray = [];
        let oneHotArray = []
        for (const c of inputText) {
            tempArray.push(uniqueChars.indexOf(c));
        }
        for (let i = 0; i < tempArray.length; i++) {
            oneHotArray[i] = new Array(vocabSize).fill(0);
            oneHotArray[i][tempArray[i]] = 1;
        }
        return oneHotArray;
    }

    getNext (values, indices) { //returns the next predicted character based on a stochastically chosen index
        if (this.craziness == 0) {
            return uniqueChars[indices[values.indexOf(Math.max(...values))]];
        }

        values = values.map(x => Math.pow(x, 10-(MathUtil.clamp(this.craziness, 0, 100)/10)))   //raise all values to the power (10-craziness/10) so they become less similar with lower craziness and more similar with higher craziness
        let nextIndex = 0;
        const add = (a, b) => a + b;
        let sum = values.reduce(add);
        let tempSum = 0;
        let rand = Math.random();
        for (let j = 0; j < indices.length; j++) {  //stochastically choose which index to use
            if (values[j]/sum + tempSum > rand) {
                nextIndex = indices[j];
                break;
            }
            tempSum += values[j]/sum;
        }
        return uniqueChars[nextIndex];  //translate index to character
    }

    genWords (args) {
        const textPromise = new Promise(resolve => {
            this.generateTextUntilCharReached(args.SEED, [' ', '\n'], args.LENGTH).then(text => {
                resolve(args.SEED+text);
                return args.SEED+text;
            })
        });
        return textPromise;
    }

    genNextWord (args) {
        const wordPromise = new Promise(resolve => {
            this.generateTextUntilCharReached(this.mostRecentChars, [' ', '\n'], 1).then(text => {
                resolve(text);
                return text;
            });
        });
        return wordPromise;
    }

    genNextSentence (args) {
        const sentencePromise = new Promise(resolve => {
            this.generateTextUntilCharReached(this.mostRecentChars, ['.', '!', '?'], 1).then(text => {
                resolve(text);
                return text;
            });
        });
        return sentencePromise;
    }

    generateTextUntilCharReached (seedStr, stopChars, len) {
        const textPromise = new Promise(resolve => {    // this function will return a promise

            //dictionary mapping user's source choice to a URL to get the file from
            let modelSourceDict = {'Dr. Seuss': 'https://raw.githubusercontent.com/Katya3141/scratch-vm/text-generation/src/extensions/scratch3_text_generation/models/seuss.json',
                                    'Shakespeare': 'https://raw.githubusercontent.com/Katya3141/scratch-vm/text-generation/src/extensions/scratch3_text_generation/models/shakespeare.json',
                                    'jokes': 'https://raw.githubusercontent.com/Katya3141/scratch-vm/text-generation/src/extensions/scratch3_text_generation/models/jokes.json',
                                    'Warriors': 'https://raw.githubusercontent.com/Katya3141/scratch-vm/text-generation/src/extensions/scratch3_text_generation/models/warriorcats.json',
                                    'Moby Dick': 'https://raw.githubusercontent.com/Katya3141/scratch-vm/text-generation/src/extensions/scratch3_text_generation/models/mobydick.json'};

            const modelSource = modelSourceDict[this.source];
            let output = '';

            tf.loadLayersModel(modelSource).then(model => {
                let nextChar = null;
                const inputDim = model.layers[0].input.shape[1];
                const paddingSize = Math.max(inputDim - seedStr.length, 0);
                let seed = this.toOneHot("*".repeat(paddingSize) + seedStr.slice(-inputDim));
                let counter = 0;

                while (counter < len && output.length < 250) {
                    let inputTensor = tf.tensor([seed]);
                    let prediction = model.predict(inputTensor);
                    let {values, indices} = tf.topk(prediction, 10);
                    let valuesArray = values.dataSync();
                    let indicesArray = indices.dataSync();

                    nextChar = this.getNext(valuesArray, indicesArray);

                    if (stopChars.includes(nextChar) && output.length > 0) {
                        counter++;
                    }
                    seed.push(this.toOneHot(nextChar)[0]);
                    seed.shift();
                    output += nextChar;
                }
                this.mostRecentChars = (this.mostRecentChars + output).slice(-20);
                resolve(output);
                return output;
            });
        });
        return textPromise;
    }

    setCraziness (args) {
        this.craziness = args.CRAZINESS;
    }

    getCraziness (args) {
        return this.craziness;
    }

    setSource (args) {
        this.source = args.SOURCE;
    }

    getSource (args) {
        return this.source;
    }
}
module.exports = Scratch3TextGeneration;
