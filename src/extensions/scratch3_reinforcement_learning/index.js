const formatMessage = require('format-message');
const nets = require('nets');

const ArgumentType = require('../../extension-support/argument-type');
const BlockType = require('../../extension-support/block-type');
const Cast = require('../../util/cast');
const MathUtil = require('../../util/math-util');
const Clone = require('../../util/clone');
const Video = require('../../io/video');
const log = require('../../util/log');


/**
 * Class for the teachable classifier blocks.
 * @constructor
 */
class Scratch3ReinforcementLearning {
    constructor (runtime) {
        this.prevState = null;
        this.prevAction = null;
        this.state = null;
        this.suggestedAction = null;
        this.epsilon = 0.5;
        this.Q = {};
        this.actions = ["up", "down", "left", "right"];

        this.DISCOUNT_FACTOR = 0.85;
        this.LEARNING_RATE = 0.6;


        this.runtime = runtime;
        this.runtime.on('ACTION_TAKEN', action => {
            this.runtime.startHats('reinforcementLearning_whenITakeAction', {
                ACTION: action
            })
        })
    }

    /**
     * @returns {object} metadata for this extension and its blocks.
     */
    getInfo () {
        return {
            id: 'reinforcementLearning',
            name: 'Reinforcement Learning',
            blocks: [
                {
                    opcode: 'whenITakeAction',
                    text: 'when I take action [ACTION]',
                    blockType: BlockType.HAT,
                    isEdgeActivated: false,
                    arguments: {
                        ACTION: {
                            type: ArgumentType.STRING,
                            menu: 'ACTION',
                            defaultValue: 'up'
                        }
                    }
                },
                // {
                //     opcode: 'addAction',
                //     text: 'add a possible action from this state [ACTION]',
                //     blockType: BlockType.COMMAND,
                //     arguments: {
                //         ACTION: {
                //             type: ArgumentType.STRING,
                //             //menu: 'ACTIONS',
                //             defaultValue: 'action 1'
                //         }
                //     }
                // },
                {
                    opcode: 'takeAction',
                    text: 'take action [ACTION]',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        ACTION: {
                            type: ArgumentType.STRING,
                            menu: 'ACTION',
                            defaultValue: 'up'
                        }
                    }
                },
                {
                    opcode: 'giveReward',
                    text: 'give reward [REWARD]',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        REWARD: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 10
                        }
                    }
                },
                {
                    opcode: 'setState',
                    text: 'set state to [STATE]',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        STATE: {
                            type: ArgumentType.STRING,
                            defaultValue: 'state 1'
                        }
                    }
                },
                {
                    opcode: 'getState',
                    text: 'state',
                    blockType: BlockType.REPORTER
                },
                {
                    opcode: 'getAction',
                    text: 'suggested action',
                    blockType: BlockType.REPORTER
                },
                {
                    opcode: 'setCuriosity',
                    text: 'set curiosity to [CURIOSITY]%',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        CURIOSITY: {
                            type: ArgumentType.NUMBER,
                            defaultValue: 50
                        }
                    }
                },
                {
                    opcode: 'getCuriosity',
                    text: 'curiosity',
                    blockType: BlockType.REPORTER
                }
            ],
            menus: {
                ACTION: 'getActionsList'
            }
        };
    }

    getActionsList () {
        return this.actions;
    }

    calculateSuggestedAction () {
        let possibleActions = Object.keys(this.Q[this.state]);
        if (possibleActions.length > 0) {
            if (Math.random() < this.epsilon) {
                this.suggestedAction = possibleActions[Math.floor(Math.random()*possibleActions.length)];
            } else {
                let maxVal = this.Q[this.state][possibleActions[0]];
                let maxActions = [possibleActions[0]];
                for (let i = 1; i < possibleActions.length; i++) {
                    if (this.Q[this.state][possibleActions[i]] > maxVal) {
                        maxVal = this.Q[this.state][possibleActions[i]];
                        maxActions = [possibleActions[i]];
                    } else if (this.Q[this.state][possibleActions[i]] === maxVal) {
                        maxActions.push(possibleActions[i]);
                    }
                }
                this.suggestedAction = maxActions[Math.floor(Math.random()*maxActions.length)];
                //this.suggestedAction = possibleActions.reduce(function(a, b){ return this.Q[this.state][a] > this.Q[this.state][b] ? a : b });
            }
        }
    }

    whenITakeAction (args) {
        return true;
    }

    addAction (args) {
        if (this.state !== null) {
            if (!(args.ACTION in this.Q[this.state])) {
                this.Q[this.state][args.ACTION] = 0;
            }
            if (!this.actions.includes(args.ACTION)) {
                this.actions.push(args.ACTION);
            }
        }

        this.calculateSuggestedAction();
    }

    takeAction (args) {
        if (!(args.ACTION in this.Q[this.state])) {
            this.Q[this.state][args.ACTION] = 0;
        }
        if (!this.actions.includes(args.ACTION)) {
            this.actions.push(args.ACTION);
        }

        this.prevAction = args.ACTION;

        this.runtime.emit('ACTION_TAKEN', args.ACTION);
    }

    giveReward (args) {
        let r = Cast.toNumber(args.REWARD);
        let oldQ = this.Q[this.prevState][this.prevAction];
        this.Q[this.prevState][this.prevAction] = (1 - this.LEARNING_RATE)*oldQ + this.LEARNING_RATE*(r + this.DISCOUNT_FACTOR*Math.max(0, ...Object.values(this.Q[this.state])));
    }

    setState (args) {
        if (!(args.STATE in this.Q)) {
            this.Q[args.STATE] = {};
        }

        this.prevState = this.state;
        this.state = args.STATE;

        this.addAction({ACTION: "up"});
        this.addAction({ACTION: "down"});
        this.addAction({ACTION: "left"});
        this.addAction({ACTION: "right"});
        
        this.calculateSuggestedAction();
        console.log(this.Q);
    }

    getState () {
        return this.state;
    }

    getAction () {
        return this.suggestedAction;
    }

    setCuriosity (args) {
        this.epsilon = args.CURIOSITY/100;
    }

    getCuriosity () {
        return this.epsilon;
    }
}
module.exports = Scratch3ReinforcementLearning;
