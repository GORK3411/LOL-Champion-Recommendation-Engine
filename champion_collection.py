import requests
import os

"""
if the champions 
"""
def  get_champions_by_id():

    path = os.path.join("data","champions_id.csv")

    if not os.path.exists(path):
        v = requests.get("https://ddragon.leagueoflegends.com/api/versions.json").json()[0]
        data = requests.get(f"https://ddragon.leagueoflegends.com/cdn/{v}/data/en_US/champion.json").json()["data"]
        id_to_name = {int(c["key"]): c["name"] for c in data.values()}
        #save as csv

    # extract the champions (id,name) and return it as a dictionary

tiers = ["GOLD","SILVER","PLATINIUM","BRONZE"]
divisions = ["I","II","III","IV"]
queue = "RANKED_SOLO_5x5"
region = "euw1"

"""
loop over the arrays "tiers" and "division" to extract players puid from /lol/league-exp/v4/entries/{queue}/{tier}/{division}
page is a query parameter.
return a list that contains the players puuid
"""
def get_players_puuid(page = 1):

    pass

"""
get the "top" most played champion of a player using /lol/champion-mastery/v4/champion-masteries/by-puuid/{encryptedPUUID}/top
remove the champions that has less than 30000 (put that value in a variable)
return a list[int] which are the champion ids 
"""

def get_top_champions(puuid, top  = 5):
    pass

"""
champions_id is a list[int].
return a dictionary{int:str} where the keys are the input keys and teh values are the values from id_name.
"""
def replace_champion_ids(champions_id):
    id_name = get_champions_by_id()
    pass