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

        this.stopListening = this.stopListening.bind(this);
        this.runtime.on('PROJECT_STOP_ALL', this.stopListening);
        this.runtime.on('COMMAND_HEARD', command => {
            this.runtime.startHats('speechCommands_whenIHear', {
                COMMAND: command
            })
        });
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
                    text: 'when I hear [COMMAND]',
                    blockType: BlockType.HAT,
                    isEdgeActivated: false,
                    arguments: {
                        COMMAND: {
                            type: ArgumentType.STRING,
                            menu: 'COMMAND',
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
                            menu: 'COMMAND',
                            defaultValue: 'go'
                        }
                    }
                },
                {
                    opcode: 'getCommand',
                    text: 'command',
                    blockType: BlockType.REPORTER
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
                COMMAND: ["go", "stop", "up", "down", "left", "right", "yes", "no", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            }
        };
    }

    whenIHear (word) {
        return true;
    }

    startListening () {
        if (!this.listening) {
            this.runtime.emitMicListening(true);
            this.listening = true;
            recognizer.ensureModelLoaded().then(resolve => {    //when the model is loaded, start listening, and continuously choose the most likely command
                recognizer.listen(result => {
                    let maxScore = Math.max(...result.scores);
                    let command = recognizer.wordLabels()[result.scores.indexOf(maxScore)];
                    if (maxScore > 0.8 && command !== '_background_noise_' && command !== '_unknown_') {  //if it recognized a word with high enough probability, set both commands to that word; otherwise reset newCommand
                        console.log(command);
                        this.newCommand = command;
                        this.mostRecentCommand = command;
                        this.runtime.emit('COMMAND_HEARD', command);
                    } else {
                        this.newCommand = '';
                    }
                }, {
                    probabilityThreshold: 0,
                    invokeCallbackOnNoiseAndUnknown: true
                });
            });
        }
    }

    stopListening () {
        if (this.listening) {
            recognizer.stopListening(); //stop the recognizer from listening
            this.runtime.emitMicListening(false);
            this.listening = false;
        }
    }

    justHeard (args) {
        return (this.mostRecentCommand === args.COMMAND);    //return the most recently heard command
    }

    getCommand () {
        return this.mostRecentCommand;
    }
}
module.exports = Scratch3SpeechCommands;
