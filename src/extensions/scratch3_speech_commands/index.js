const formatMessage = require('format-message');
const nets = require('nets');

const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const Cast = require('../../util/cast');
const MathUtil = require('../../util/math-util');
const Clone = require('../../util/clone');
const log = require('../../util/log');
const tf = require('@tensorflow/tfjs');
const speechCommandsModel = require('@tensorflow-models/speech-commands');

const recognizer = speechCommandsModel.create('BROWSER_FFT');

const serverTimeoutMs = 10000;


/**
 * Class for the speech commands blocks.
 * @constructor
 */
class Scratch3SpeechCommands {
    constructor (runtime) { //keep track of the previous command, and whether the extension is currently listening for commands
        this.runtime = runtime;
        this.newCommand = '';   //resets every time there's only _background_noise_ or _unknown_
        this.mostRecentCommand = '';    //does not reset until there's a new command
        this.listening = false;

        this.runtime.on('PROJECT_STOP_ALL', this.stopListening.bind(this));
    }

    
    /**
     * @returns {object} metadata for this extension and its blocks.
     */
    getInfo () {
        return {
            id: 'speechCommands',
            name: 'Speech Commands',
            blocks: [
                {
                    opcode: 'whenIHear',
                    text: 'when I hear [WORD]',
                    blockType: BlockType.HAT,
                    arguments: {
                        WORD: {
                            type: ArgumentType.STRING,
                            menu: 'WORD',
                            defaultValue: 'go'
                        }
                    }
                },
                {
                    opcode: 'justHeard',
                    text: 'just heard [COMMAND]?',
                    blockType: BlockType.BOOLEAN,
                    arguments: {
                        COMMAND: {
                            type: ArgumentType.STRING,
                            menu: 'WORD',
                            defaultValue: 'go'
                        }
                    }
                },
                {
                    opcode: 'startListening',
                    text: 'start listening',
                    blockType: BlockType.COMMAND
                },
                {
                    opcode: 'stopListening',
                    text: 'stop listening',
                    blockType: BlockType.COMMAND
                }
            ],
            menus: {
                WORD: ["go", "stop", "up", "down", "left", "right", "yes", "no", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            }
        };
    }

    whenIHear (word) {
        this.startListening();
        let toReturn = this.newCommand == word.WORD;  //return true if the previous command was the word being listened for, false otherwise
        this.newCommand = '';
        return toReturn;
    }

    startListening () {
        if (!this.listening) {
            this.runtime.emitMicListening(true);
            this.listening = true;
            recognizer.ensureModelLoaded().then(resolve => {    //when the model is loaded, start listening, and continuously choose the most likely command
                this.newCommand = '';
                recognizer.listen(result => {
                    command = recognizer.wordLabels()[result.scores.indexOf(Math.max(...result.scores))];
                    console.log(command);
                    this.newCommand = command;
                    this.mostRecentCommand = command;
                }, {
                    probabilityThreshold: 0,
                    invokeCallbackOnNoiseAndUnknown: true
                });
            });
        }
    }

    stopListening () {
        if (this.listening) {   //if currently listening, stop
            this.runtime.emitMicListening(false);
            recognizer.stopListening();
            this.listening = false;
        }
    }

    justHeard (args) {
        return (this.mostRecentCommand == args.COMMAND);    //return the most recently heard command
    }

    resetCommand () {
        this.mostRecentCommand = '';    //reset the command to an empty string
        this.newCommand = '';
    }
}
module.exports = Scratch3SpeechCommands;
