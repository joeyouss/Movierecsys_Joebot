
# 25/05 -> understanding Discord API (creating a simple bot in sampletest.py)
# 26/05 -> reading collborative filtering and its similarity measures
# 28/05 -> choosing between cosine similarity and pearson since i used cosine for the last time in collaboarative
# 29/05 -> integrating joke API in bot to see its working 
# 2/06 -> creating collaborative model
# 3/06 -> not working, creating model again with pearson
# 4/06 -> try content based filtering using cosine similarity 
# 5/06 -> [update] laptop freeze at heavy function uplifiting in cosine
# 7/06 -> flask integration for running server 
# 8/06 -> add helpline numvers in view of coding
# 9/06-> apply pearson and ddefine bot commands
# 10/06 -> [update] error in pearson formula-> debug
# 11/06 -> hosting using repl it [update] -> error
# 12/06 -> error in server render jinja templating
# 15/06 0> bot is not user oriented -> make it functional for programming purposes


# --------------------------
# # loopholes
# -> not a running server
# -> not an sync movie recsys


import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import discord
import json
import os
import requests  
from discord.ext import commands
import random
TOKEN = ""

bot = commands.Bot(command_prefix=">")
# @bot.command
@bot.event
async def on_ready():
    print('Bot is ready')


# ----------------part 1 - reading datasets=---------
mv_data = pd.read_csv('bot_datasets/movies.csv')
r_data = pd.read_csv('bot_datasets/ratings.csv')
# -----------------part 2- finding release years (since it is embedded)-------
mv_data['year'] = mv_data.title.str.extract('(\(\d\d\d\d\))',expand=False)
mv_data['year'] = mv_data.year.str.extract('(\d\d\d\d)',expand=False)
mv_data['title'] = mv_data.title.str.replace('(\(\d\d\d\d\))', '')
mv_data['title'] = mv_data['title'].apply(lambda x: x.strip())
# ----------------part 3 - dropping unanted columns---------------------
mv_data = mv_data.drop('genres',1)
r_data = r_data.drop('timestamp', 1)

@bot.command(pass_context= True)
async def recsys(ctx,keyword):
# ----------------part 4 -> taking user input
    input_by_user = []
    input_by_user.append({'title':keyword, 'rating':5})

    input_by_user_df = pd.DataFrame(input_by_user)
# ---------------please enter the movie and ratings
# ----------------part 5 -> working for making the input usable for my bot-----------

    user_input_id = mv_data[mv_data['title'].isin(input_by_user_df['title'].tolist())]
    input_by_user_df = pd.merge(user_input_id, input_by_user_df)
    input_by_user_df = input_by_user_df.drop('year', 1)


# ----------------part 6 -> finding users who have actuslly seen this movie for my bot
    watched_users = r_data[r_data['movieId'].isin(input_by_user_df['movieId'].tolist())]
# part 7 -> creating user groups for collaborative filtering
    users_group = watched_users.groupby(['userId'])
    users_group = sorted(users_group, key = lambda ok : len(ok[1]), reverse = True)
# part 8 -> applying pearson correlation - taking top 50 due to system limitations of my laptop (update: try extending it)
    users_group = users_group[0:100]
    pearsonCorrelationDict= {}
    
    for name, group in users_group:
        group = group.sort_values(by='movieId')
        input_by_user_df = input_by_user_df.sort_values(by='movieId')
        n = len(group)
        ok_df = input_by_user_df[input_by_user_df['movieId'].isin(group['movieId'].tolist())]
        ratings = ok_df['rating'].tolist()
        group_ok = group['rating'].tolist()
        den1 = sum([i**2 for i in ratings]) - pow(sum(ratings),2)/float(n)
        den2 = sum([i**2 for i in group_ok]) - pow(sum(group_ok),2)/float(n)
        numerator = sum( i*j for i, j in zip(ratings, group_ok)) - sum(ratings)*sum(group_ok)/float(n)
    
    #If the denominator is different than zero, then divide, else, 0 correlation.
        if den1 != 0 and den2 != 0:
            pearsonCorrelationDict[name] = numerator/sqrt(den1*den2)
        else:
            pearsonCorrelationDict[name] = 0

# cjhecking checking if it is working right
# print(pearsonCorrelationDict.items())
# --------------------part 9 -> 
    resultant_df = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
    resultant_df.columns = ['similarityIndex']
    resultant_df['userId'] = resultant_df.index
    resultant_df.index = range(len(resultant_df))

#step 10 -> getting top users similar to out user
    top_u = resultant_df.sort_values(by='similarityIndex', ascending=False)[0:50]
    top_ratings=top_u.merge(r_data, left_on='userId', right_on='userId', how='inner')
    top_ratings['weightedRating'] = top_ratings['similarityIndex']*top_ratings['rating']

# ------------------------------

    user_top_rating = top_ratings.groupby('movieId').sum()[['similarityIndex','weightedRating']]
    user_top_rating.columns = ['sum_similarityIndex','sum_weightedRating']
    final_df = pd.DataFrame()
    final_df['weighted average recommendation score'] = user_top_rating['sum_weightedRating']/user_top_rating['sum_similarityIndex']
    final_df['movieId'] =user_top_rating.index
    final_df = final_df.sort_values(by='weighted average recommendation score', ascending=False)
# ----------------------------------------------
    mm = mv_data.loc[mv_data['movieId'].isin(final_df.head(7)['movieId'].tolist())]

    ll = mm.values.tolist()
    for i in ll:
        res = i[1] +" released in -"+i[2]
        await ctx.send(res)
        
    

# ---------------------------------------------------------helplines (an add on)------
helplines = ["Family Violence Prevention Center 1-800-313-1310",
             "National Sexual Assault Hotline 1-800-656-HOPE (4673)",
             "Drug Abuse National Helpline 1-800-662-4357",
             "American Cancer Society 1-800-227-2345",
             "Eating Disorders Awareness and Prevention 1-800-931-2237",
             "GriefShare 1-800-395-5755",
             "Suicide Hotline 1-800-SUICIDE (784-2433)"]

@bot.command(aliases=['h'], help="Get all helpline numbers related to mental health")
async def checkkk(ctx):
    for i in helplines:
        res = i
        await ctx.send(res)

@bot.command(aliases=['depression'], help="Get helpline number to tackle depression and suicidal thoughts")
async def suicide(ctx):
    await ctx.send(helplines[6])

@bot.command(aliases=['violent'], help="Get helpline number to tackle violence")
async def violence(ctx):
    await ctx.send(helplines[0])

@bot.command(aliases=['drugs'], help="Get helpline number to tackle drug addiction")
async def drugabuse(ctx):
    await ctx.send(helplines[2])

@bot.command(aliases=['assault'], help="Get helpline number to tackle assaults")
async def sexualAssault(ctx):
    await ctx.send(helplines[1])

@bot.command(aliases=['ILLNESS'], help="Get helpline number to tackle illness")
async def cancer(ctx):
    await ctx.send(helplines[3])

@bot.command(aliases=['sorrow'], help="Get helpline number to tackle grief")
async def grief(ctx):
    await ctx.send(helplines[5])

@bot.command(aliases=['eating'], help="Get helpline number to tackle eating disorder")
async def eatingdisorder(ctx):
    await ctx.send(helplines[4])
# ------------------------------------------------------------
greet = [
    "Welcome !. What can I do for you?",
    "Hi!!",
    "I am so excited to have you here!",
    "Hello there, I am an MLH bot here for your help",
    "No I am sleeping right now. Just kidding !! how may I help you?"
]

starter_encouragements = [
    "At times you may feel like this and I understand that, but it won't be for long I can assure you",
    "Trying going for a walk and disconnecting with your surroundings and listen to music. The best way to cope up is to take a break",
    "I hear you, MLH is there for you. Please talk to your pod leader and tell him in detail what problems are you facing in, he/she will be dhtere for you like we all are",
    "You are doing great. If it counts, I am proud of you ! Don't feel like you cannot do better, you ofcourse can!"
]

copeup=[
    "I can understand totally. But do you know how can you cope up?. Let me help you : 1. Divide your time into blocks"
    "and make sure you are being generous to yourself."
    "2. Make sure you drink enough water and go out for a walk when your mind gets cluttered."
    "3. A little music does not hurt anyone."
    "4. We as coders feel as impostors all the time, don't worry just make sure you have faith on yourself",
    "I can totally understand, have you ever tried the pomodore technique.? Read about it here - https://en.wikipedia.org/wiki/Pomodoro_Technique#:~:text=The%20Pomodoro%20Technique%20is%20a%20time%20management%20method,25%20minutes%20in%20length%2C%20separated%20by%20short%20breaks. "
]

# --------------------------------------------------------------
@bot.command(pass_context= True)
async def joke(ctx):
    url = "https://v2.jokeapi.dev/joke/Programming,Miscellaneous,Christmas?blacklistFlags=nsfw,political,racist,sexist,explicit&type=twopart"
    
    response = requests.get(url)
    json_data = json.loads(response.text)
    jokes = " "
    jokes = json_data["setup"]+" - "+json_data["delivery"]
    await ctx.send(jokes)
# async def activate(ctx):
#     await ctx.send("Hi master, I am ready, please enter the name of movie you would like to be recommended for today")
#     def check(msg):
#         return msg.author==ctx.author and msg.channel == ctx.channel
#     msg = await bot.wait_for("message", check = check)
#     
#     input_by_user.append({'title':msg, 'rating':4.5})

     
# -------------------------------------------
@bot.command(pass_context= True)
async def quote(ctx):
    response = requests.get("https://zenquotes.io/api/random")
    json_data = json.loads(response.text)
    quote = json_data[0]['q'] + " -" + json_data[0]['a']
    # return(quote)
    await ctx.send(quote)
# --------------------------------------------------------
greetings = ["hi", "hello","how are you", "whats up","hey"]
sad_words = ["sad", "depressed", "unhappy", "angry", "miserable", "stressed", "hopeless", "unhappy","worthless"]
problems = ["time management", "busy", "impostor", "coding"]
suicide = ["kill myself", "kill yourself","suicidal","death"]

# --------------------------------listing all commands written
@bot.command(pass_context=True)
async def all(ctx):
    res=  ["Hi, These are the list of commands available currently:",
         ">all - gives a list of all commands (you know about it since you used it to see this!)",
         ">quote - A quote to enlighten you !",
         ">recsys - Want me to recommend a movie to you? just type in >rec [name of movie] and I will bring out something from my intelligence for you!",
         ">joke - Sometimes we are just in need of a small joke. Don't worry, I am just testing my REST API!"
         ]
    for i in res:
        await ctx.send(i)
  
# -----------------------------------------------
@bot.event
async def on_message(message):
    await bot.process_commands(message)
    msg = message.content
    flag = False
    if message.author == bot.user:
        return 
    if any(word in msg for word in greetings):
        gify = True
        response = random.choice(greet)
        await message.channel.send(response)

    if any(word in msg for word in sad_words):
        flag = True
        response = random.choice(starter_encouragements)
        await message.channel.send(response)

    if any(word in msg for word in problems):
        flag = True
        response = random.choice(copeup)
        await message.channel.send(response)

    # if(flag):
    #     r1 = "And just to brighten your day, here is a joke for you"
    #     await message.channel.send(r1)
    #     r2 = joke()
    #     await message.channel.send(r2)
    
    # if "activate
    
bot.run(TOKEN)
