import re
import random
from torch import rand
from intentClassification import ic_model 
from pricePrediction import pp_model
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
pd.set_option('display.max_colwidth', 1000)

test_ds = load_dataset('craigslist_bargains', split= 'validation')
test_ds = pd.DataFrame(test_ds)
test_ds.rename(columns={'utterance':'bargain_convo','dialogue_acts':'intent'}, inplace=True)
test_ds = test_ds.loc[6,:]

def priceExtraction(value):
    try:
        price = re.findall('\$\d+', value)
        return int(price)
    except:
        return 0

def getPriceLimit():
    price_limit = np.load('price_limit.npy')
    return price_limit[0], price_limit[1]

def getBuyerBids():
    buyer_timeline = np.load('buyer_timeline.npy')
    buyer_bids = [x[1] for x in buyer_timeline if x[1] != 0]
    buyer_bids = buyer_bids[1:]
    return buyer_bids

def getBuyerIntents():
    buyer_timeline = np.load('buyer_timeline.npy')
    buyer_intents = [x[0] for x in buyer_timeline]
    buyer_intents = buyer_intents[1:]
    return buyer_intents

def getBotBids():
    bot_timeline = np.load('bot_timeline.npy')
    bot_bids = [x[1] for x in bot_timeline if x[1] != 0]
    bot_bids = bot_bids[1:]
    return bot_bids

def getBotIntents():
    bot_timeline = np.load('bot_timeline.npy')
    bot_intents = [x[0] for x in bot_timeline]
    bot_intents = bot_intents[1:]
    return bot_intents

def saveIntentIntoBuyerTimeline(data):
    #saving current buyer's and bot's intent and bids in file intent_timeline.npy
    buyer_timeline = np.load('buyer_timeline.npy')
    new_timeline = np.append(buyer_timeline, [data], axis=0)
    np.save('buyer_timeline.npy', new_timeline)
    print('buyer_timeline:')
    print(new_timeline)

def saveIntentIntoBotTimeline(data):
    #saving current buyer's and bot's intent and bids in file intent_timeline.npy
    bot_timeline = np.load('bot_timeline.npy')
    new_timeline = np.append(bot_timeline, [data], axis=0)
    np.save('bot_timeline.npy', new_timeline)
    print('bot_timeline:')
    print(new_timeline)

def discountedAmount(buyer_bid):
    discount = pp_model.max_discount_predict(getBuyerIntents())
    upperLimit, lowerLimit = getPriceLimit()
    bot_bid = (100 - discount[0]) * 0.01 * upperLimit
    if bot_bid == buyer_bid: 
        return int(buyer_bid)
    print("price prediction:", bot_bid)
    bot_bid = bot_bid if bot_bid > buyer_bid else buyer_bid
    bot_bid = bot_bid if bot_bid > lowerLimit else (upperLimit + lowerLimit)//1.95
    bot_all_bids = getBotBids()
    if len(bot_all_bids) > 0:
        last_bid = int(bot_all_bids[-1])
        bot_bid = bot_bid if bot_bid < last_bid else (last_bid + lowerLimit)//1.95
    saveIntentIntoBotTimeline(['counter-price',int(bot_bid)])
    print("price approximation:", bot_bid)
    return int(bot_bid)

def Intentagree(buyer_bid):
    #once the deal is done, reset previous intents of ongoing conversation
    if buyer_bid == 0:
        all_bot_bids = getBotBids()
        if len(all_bot_bids) == 0:
            upperLimit, lowerLimit = getPriceLimit()
            buyer_bid = upperLimit   
        else:
            buyer_bid = all_bot_bids[-1]
    saveIntentIntoBuyerTimeline(['agree',buyer_bid])
    return 'agree', buyer_bid

def Intentintro(buyer_bid):
    saveIntentIntoBuyerTimeline(['intro',buyer_bid])
    return 'intro', None

def Intentinquiry(buyer_bid):
    saveIntentIntoBuyerTimeline(['inquiry',buyer_bid])
    return 'inquiry', None

def Intentinitprice(buyer_bid):
    buyer_intents = getBuyerIntents()
    count = 0
    for x in buyer_intents: 
        if x in ['counter-price','init-price']: 
            count+=1
    if count > 4:
        print("log: Vague counter-pricing. Switching to Intent-disagree for response")
        return 'disagree', None

    bot_bid = discountedAmount(buyer_bid)
    if bot_bid == buyer_bid or ((bot_bid - buyer_bid)/bot_bid)*100 < 1.2:
        print("log: Equal bids. Switching to Intent-agree for response")
        return Intentagree(buyer_bid)
    # if bot_bid == buyer_bid:
    #     return eval("Intentagree" + "({})".format(buyer_bid))
    # if bot_bid < buyer_bid:
    #     return eval("Intentagree" + "({})".format(buyer_bid))
    saveIntentIntoBuyerTimeline(['init-price',buyer_bid])
    return 'init-price', bot_bid

def Intentcounterprice(buyer_bid):
    buyer_intents = getBuyerIntents()
    count = 0
    for x in buyer_intents: 
        if x in ['counter-price','init-price']: 
            count+=1
    if count > 4:
        print("log: Vague counter-pricing. Switching to Intent-disagree for response")
        return 'disagree', None
    bot_bid = discountedAmount(int(buyer_bid))
    if bot_bid == buyer_bid or ((bot_bid - buyer_bid)/bot_bid)*100 < 1.2:
        print("log: Equal bids. Switching to Intent-agree for response")
        return Intentagree(buyer_bid)
    # if bot_bid == buyer_bid:
    #     return eval("Intentagree" + "({})".format(buyer_bid))
    # if bot_bid < buyer_bid:
    #     return eval("Intentagree" + "({})".format(buyer_bid))
    saveIntentIntoBuyerTimeline(['counter-price',buyer_bid])
    return 'counter-price', bot_bid

def Intentdisagree(buyer_bid):
    buyer_intents = getBuyerIntents() 
    count = 0
    for x in buyer_intents: 
        if x in ['counter-price','init-price']: count+=1
    saveIntentIntoBuyerTimeline(['disagree',buyer_bid])
    # if count > random.randrange(4,6):
    if count > 1:
        return 'disagree', None
    else:
        last_buyer_bid = getBuyerBids()
        buyer_bid = buyer_bid if buyer_bid > 0 else last_buyer_bid[-1]
        return Intentcounterprice(int(buyer_bid))

def Intentunknown(buyer_bid):
    saveIntentIntoBuyerTimeline(['unknown',buyer_bid])
    return 'unknown', None
    
def decisionEngine(text):
    #function call for intentClassification from file ic_model.py
    buyer_intent = ic_model.predict_intent(text)
    buyer_intent = 'counterprice' if buyer_intent == 'counter-price' else buyer_intent
    buyer_intent = 'initprice' if buyer_intent == 'init-price' else buyer_intent

    #fetching discret price-range or buyers-bid from the user-input text i.e (rangeMin,rangeMax) or (singleValue,0) 
    #e.g.(100,200), (100,0)
    
    buyer_bid = priceExtraction(text)

    all_buyer_bids = getBuyerBids()
    if all_buyer_bids.count(buyer_bid) >= 2: return Intentdisagree(buyer_bid)
        # upperLimit, lowerLimit = getPriceLimit()
        # buyer_all_bids = getBuyerBids()
        # last_bid = buyer_all_bids[-1]
        # second_last_bid = buyer_all_bids[-2]
        # if all(x < lowerLimit for x in [second_last_bid ,last_bid, buyer_bid]):
        #     return eval("Intentdisagree" + "({})".format(0))

    # print(buyer_bid)
    #calling distint functions according to the buyer_intent
    return eval("Intent" + str(buyer_intent) + "({})".format(buyer_bid))
