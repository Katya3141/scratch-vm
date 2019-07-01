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

const coords = {"Cambridge, MA": [42.3736, -71.1097], "Chicago, IL": [41.8781, -87.6298], "Seattle, WA": [47.6062,-122.3321]};


/**
 * Class for the speech commands blocks.
 * @constructor
 */
class Scratch3Weather {
    constructor (runtime) { //keep track of the previous command, and whether the extension is currently listening for commands
        [this.latitude, this.longitude] = coords["Cambridge, MA"];
    }

    
    /**
     * @returns {object} metadata for this extension and its blocks.
     */
    getInfo () {
        return {
            id: 'weather',
            name: 'Weather',
            blocks: [
                {
                    opcode: 'setLocation',
                    text: 'set location to [LOCATION]',
                    blockType: BlockType.COMMAND,
                    arguments: {
                        LOCATION: {
                            type: ArgumentType.STRING,
                            menu: 'LOCATION',
                            defaultValue: 'Cambridge, MA'
                        }
                    }
                },
                {
                    opcode: 'temperature',
                    text: 'temperature',
                    blockType: BlockType.REPORTER
                },
                {
                    opcode: 'conditions',
                    text: 'weather conditions',
                    blockType: BlockType.REPORTER
                },
                {
                    opcode: 'windSpeed',
                    text: 'wind speed',
                    blockType: BlockType.REPORTER
                }
            ],
            menus: {
                LOCATION: ["Cambridge, MA", "Chicago, IL", "Seattle, WA"]
            }
        };
    }

    setLocation (args) {
        [this.latitude, this.longitude] = coords[args.LOCATION];
    }

    temperature () {
        return this.weather('temperature');
    }

    conditions () {
        return this.weather('shortForecast');
    }

    windSpeed () {
        return this.weather('windSpeed');
    }

    weather (query) {
        let zoneURL = 'https://api.weather.gov/points/'
        zoneURL += this.latitude + ',' + this.longitude

        const weatherPromise = new Promise(resolve => {
            nets({
                url: zoneURL,
                timeout: serverTimeoutMs
            }, (err, res, body) => {
                if (err) {
                    log.warn(`error fetching result! ${res}`);
                    resolve(err);
                    return err;
                }
                const forecastURL = JSON.parse(body).properties.forecast;
                nets({
                    url: forecastURL,
                    timeout: serverTimeoutMs
                }, (err, res, body) => {
                    if (err) {
                        log.warn(`error fetching result! ${res}`);
                        resolve(err);
                        return err;
                    }
                    const response = eval('JSON.parse(body).properties.periods[0].' + query);
                    resolve(response);
                    return response;
                })
            })
        });
        return weatherPromise;
    }
}
module.exports = Scratch3Weather;
