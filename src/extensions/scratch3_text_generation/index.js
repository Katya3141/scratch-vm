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
        this.craziness = 8;
        this.mostRecentChars = '';
        this.source = 'Dr. Seuss';
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
                    opcode: 'setCraziness',
                    text: 'set craziness to [CRAZINESS]',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        CRAZINESS: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 8
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

    getNext (values, indices, craziness) { //returns the next predicted character based on a stochastically chosen index
        if (craziness == 0) {
            return uniqueChars[indices[values.indexOf(Math.max(...values))]];
        }

        values = values.map(x => Math.pow(x, 10 - craziness))   //raise all values to the power (10 - craziness) so that they become more or less similar
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

    processInput (model, seed) {    //ensure that the input seed is the right size to be used as input to the model

        let inputDim = model.layers[0].input.shape[1];

        while (seed.length < inputDim) {    //pad the beginning of the input with stars
            seed = '*' + seed;
        }
        seed = seed.slice(seed.length-inputDim);    //only consider the last inputDim characters

        return seed;
    }

    genWords (args) {
        return this.generateText(args.LENGTH, args.SEED);
    }

    genNextWord (args) {
        const wordPromise = new Promise(resolve => {
            this.generateText(2, this.mostRecentChars).then(text => {
                nextWord = text.slice(Math.max(text.lastIndexOf(' '), text.lastIndexOf('\n')));
                resolve(nextWord);
                return nextWord;
            });
        });
        return wordPromise;
    }

    generateText (length, seedStr) {
        const textPromise = new Promise(resolve => {    // this function will return a promise

            //dictionary mapping user's source choice to a URL to get the file from
            let modelSourceDict = {'Dr. Seuss': 'https://raw.githubusercontent.com/Katya3141/scratch-vm/text-generation/src/extensions/scratch3_text_generation/models/seuss.json',
                                    'Shakespeare': 'https://raw.githubusercontent.com/Katya3141/scratch-vm/text-generation/src/extensions/scratch3_text_generation/models/shakespeare.json',
                                    'jokes': 'https://raw.githubusercontent.com/Katya3141/scratch-vm/text-generation/src/extensions/scratch3_text_generation/models/jokes.json',
                                    'Warriors': 'https://raw.githubusercontent.com/Katya3141/scratch-vm/text-generation/src/extensions/scratch3_text_generation/models/warriorcats.json',
                                    'Moby Dick': 'https://raw.githubusercontent.com/Katya3141/scratch-vm/text-generation/src/extensions/scratch3_text_generation/models/mobydick.json'};

            const endChars = [':', ',', ';', '-', '/']; //when generating strings, don't end with any of these characters

            let craziness = MathUtil.clamp(this.craziness, 0, 10);  //initialize craziness and which model to use, and initialize output to the seed string
            let modelSource = modelSourceDict[this.source];
            let output = seedStr;

            tf.loadLayersModel(modelSource).then(model => { //load model

                seedStr = this.processInput(model, seedStr);  //make sure the input has the right dimensions

                let w = 0;  //0 words so far, not between words
                let betweenWords = false;

                while (w < length) { //while more words need to be generated
                    let seed = this.toOneHot(seedStr);

                    let inputTensor = tf.tensor([seed]);    //create a tensor to use as input to the model
                    let prediction = model.predict(inputTensor);    //make a prediction
                    let {values, indices} = tf.topk(prediction, 10);    //create arrays of indices (can be translated to characters) and values (their probabilities) based on the prediction
                    let valuesArray = values.dataSync();
                    let indicesArray = indices.dataSync();

                    let nextChar = this.getNext(valuesArray, indicesArray, craziness); //choose a next character

                    if (!betweenWords && (nextChar == ' ' || nextChar == '\n')) {   //find spaces between words
                        w++;
                        betweenWords = true;
                    } else if (!(nextChar == ' ' || nextChar == '\n')) {
                        betweenWords = false;
                    }

                    if (w < length) {
                        seedStr = seedStr.slice(1);   //make a new seed which includes the new character
                        seedStr += nextChar;

                        if (w < length - 1 || !endChars.includes(nextChar)) {    //don't end the string with any of the characters in endChars
                            output += nextChar; //add the new character to the output
                        }
                    }
                }
                this.mostRecentChars = output.slice(-20);
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
