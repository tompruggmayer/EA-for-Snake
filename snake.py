import random
import time
from KNN import *
import pickle
from getMove import getMoveSnake
from datetime import datetime
from random import randint

WINDOW = 800
TILE_SIZE = 40
FELDGROESSE = 20
snake_dir = (0, 0)
time1, time_step = 0, 20
dirs = {'HOCH': 1, 'RUNTER': 1, 'LINKS': 1, 'RECHTS': 1}
#Anzahl der Neuronen im Input Layer
INPUTLAYERNEURONS = 16
#Anzahl der Neuronen im Hidden Layer
HIDDENLAYERNEURONS1 = 11
#Anzahl der Neuronen im Output Layer
OUTPUTLAYERNEURONS = 4
#wieviele individuen pro generation
GENERATIONSIZE = 1000
#prozentsatz der der besten individuen einer generation, die sich vermehren startwert
BESTSIZE_START = 0.1
#prozentsatz der der besten individuen einer generation, die sich vermehren endwert
BESTSIZE_ENDE = 0.1
#wieviele generationen lang wird simuliert
NUMBEROFGENERATIONS = 20000
#wieviel prozent random anteil an kombinierten gewicht zum Start der Simulation
MUTATIONSFAKTOR_START = 0.15
#wieviel prozent random anteil an kombinierten gewicht zum Ende der Simulation
MUTATIONSFAKTOR_ENDE = 0.15
#wieviel prozent von snakes in einer generation mutieren zum Start der Simulation
MUTATIONSGROESSE_START = 0.15
#wieviel prozent von snakes in einer generation mutieren zum Ende der Simulation
MUTATIONSGROESSE_ENDE = 0.15
#wieviele timesteps haben die snakes zeit pro game um einen apfel zu fressen
TIMESTEPS = 150
#wie lange warte pro frame
WAITTIME = 0.0
#simulationszeit in Stunden
SIMULATIONSZEIT_STUNDEN = 3
#simulationszeit in Sekunden
SIMULATIONSZEIT = 60*60*SIMULATIONSZEIT_STUNDEN

#PATH = "SIMULATIONSERGEBNISSE/wechselnder_Mutationsfaktor/"
#PATH = "SIMULATIONSERGEBNISSE/halbe_generationSize/"
#PATH = "SIMULATIONSERGEBNISSE/wechselnde_Mutationsgroesse/"
#PATH = "SIMULATIONSERGEBNISSE/normal/"
#PATH = "SIMULATIONSERGEBNISSE/kleine_Mutationsgroesse/"
#PATH = "SIMULATIONSERGEBNISSE/wechselnde_BestSize/"
#PATH = "SIMULATIONSERGEBNISSE/Zeitabhängig/normal-fitnessfunktion2/"
#PATH = "SIMULATIONSERGEBNISSE/Zeitabhängig/kleine_Mutationsgroesse/"
#PATH = "SIMULATIONSERGEBNISSE/Zeitabhaengig/wechselnde_Mutationsgroesse/"
#PATH = "SIMULATIONSERGEBNISSE/Zeitabhaengig/mittlere_Mutationsgroesse/"
#PATH = "SIMULATIONSERGEBNISSE/Zeitabhaengig/halbe_generationSize/"
#PATH = "SIMULATIONSERGEBNISSE/Zeitabhaengig/15000_generationSize/"
#PATH = "SIMULATIONSERGEBNISSE/Zeitabhaengig/0.2_Mutationsgroesse/"
#PATH = "SIMULATIONSERGEBNISSE/Zeitabhaengig/0.0_Mutationsgroesse/"
#PATH = "SIMULATIONSERGEBNISSE/Zeitabhaengig/0.5_Mutationsgroesse/"
PATH = ""



IDENTIFIER = str(randint(1000000, 9999999))

MUTATIONSART = "MÜNZWURF"
#MUTATIONSART = "ONE-POINT-CROSSOVER"

Leeres_Feld = [[0 for _ in range(FELDGROESSE+2)] for _ in range(FELDGROESSE+2)]
for i in range(len(Leeres_Feld)):
    for j in range(len(Leeres_Feld[i])):
        if i==0 or i==FELDGROESSE+1:
            Leeres_Feld[i][j]=8
        if j==0 or j==FELDGROESSE+1:
            Leeres_Feld[i][j]=8

Leere_Felder = []
for i in range(1,FELDGROESSE):
    for j in range(1,FELDGROESSE):
        Leere_Felder.append((i,j))

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    OKCYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# zufällige Rückgabe einer der beiden Gewichte    
def kombinieren3(gewicht1,gewicht2):
    auswahl = random.uniform(0,1)
    if auswahl > 0.5:
        return gewicht1
    else: 
        return gewicht2
    

def mutieren(snakes,zeitstempel):
    random.shuffle(snakes)
    # Berechnung der aktuellen mutationsgröße als linearer verlauf vom start zum endwert
    aktuellerMutationsgroesse = ((MUTATIONSGROESSE_ENDE-MUTATIONSGROESSE_START)/(SIMULATIONSZEIT))*zeitstempel + MUTATIONSGROESSE_START

    mutanten = snakes[0:int(len(snakes)*aktuellerMutationsgroesse)]
    snakes = snakes[int(len(snakes)*aktuellerMutationsgroesse):len(snakes)]

    # Berechnung des aktuellen mutationsfaktors als linearer verlauf vom start zum endwert
    aktuellerMutationsFaktor = ((MUTATIONSFAKTOR_ENDE-MUTATIONSFAKTOR_START)/(SIMULATIONSZEIT))*zeitstempel + MUTATIONSFAKTOR_START
    mutierte = []
    for mutant in mutanten:
        hiddenGewichteMutant = mutant.returnHiddenNeuronsGewichte()
        outputGewichteMutant = mutant.returnOutputNeuronsGewichte()
        hiddenGewichteMutantNEU = []
        outputGewichteMutantNEU = []
        for gewichteNeuron in hiddenGewichteMutant:
            tempGewichteHidden = []
            for gewicht in gewichteNeuron:
                mutationsRandom = random.uniform(0,1)
                if mutationsRandom<aktuellerMutationsFaktor:
                    tempGewichteHidden.append(random.uniform(-1,1))
                else:
                    tempGewichteHidden.append(gewicht)
            hiddenGewichteMutantNEU.append(tempGewichteHidden)
        for gewichteNeuron in outputGewichteMutant:
            tempGewichteOutput = []
            for gewicht in gewichteNeuron:
                mutationsRandom = random.uniform(0,1)
                if mutationsRandom<aktuellerMutationsFaktor:
                    tempGewichteOutput.append(random.uniform(-1,1))
                else:
                    tempGewichteOutput.append(gewicht)
            outputGewichteMutantNEU.append(tempGewichteOutput)
        mutierte.append(initNN2(SnakeBrainNeu(),INPUTLAYERNEURONS,hiddenGewichteMutantNEU,outputGewichteMutantNEU))
    snakes = snakes + mutierte
    random.shuffle(snakes)
    return snakes


def onePointCrossover(femaleSnake, maleSnake):
    gewichteFemaleHidden1 = femaleSnake.returnHiddenNeuronsGewichte()
    gewichteFemaleOutput = femaleSnake.returnOutputNeuronsGewichte()
    gewichteMaleHidden1 = maleSnake.returnHiddenNeuronsGewichte()
    gewichteMaleOutput = maleSnake.returnOutputNeuronsGewichte()

    gewichteKombiniertHidden = gewichteMaleHidden1[0:int(0.5*len(gewichteMaleHidden1))] + gewichteFemaleHidden1[int(0.5*len(gewichteFemaleHidden1)):len(gewichteFemaleHidden1)]
    gewichteKombiniertOutput = gewichteMaleOutput[0:int(0.5*len(gewichteMaleOutput))] + gewichteFemaleOutput[int(0.5*len(gewichteFemaleOutput)):len(gewichteFemaleOutput)]
    #print("GEWICHTEKOMBINIERTHIDDEN: " + str([temp for temp in gewichteKombiniertHidden]))
    #print(len(gewichteKombiniertOutput))
    return gewichteKombiniertHidden,gewichteKombiniertOutput


#newGeneration: gibt neue Generation von snakes wieder, entstanden aus den besten der vorherigen Generation 
# Unterteilt die besten x% in male und female, bei dem jeder mit jedem kombiniert wird
# len(output) = (bestSize*generationSize/2)**2
def newGeneration(snakes,zeitStempel):
    #get die besten x% der vorherigen generation
    snakes.sort(key=lambda a: a[1])
    aktuelleBestSize = ((BESTSIZE_ENDE-BESTSIZE_START)/SIMULATIONSZEIT)*zeitStempel + BESTSIZE_START
    snakes = snakes[int((1-aktuelleBestSize)*len(snakes)):len(snakes)]            #die besten x%

    random.shuffle(snakes)
    #aufteilen der snakes in male und female
    femaleSnakes = snakes[int(0.5*len(snakes)):len(snakes)]
    maleSnakes = snakes[0:int(0.5*len(snakes))]
    femaleSnakes.sort(key=lambda a: a[1])
    maleSnakes.sort(key=lambda a: a[1])

    #erstellen einer neuen generation von snakes
    newSnakes = []
    [newSnakes.append(SnakeBrainNeu()) for _ in range(2*GENERATIONSIZE)]
    count = 0
    for i in range(len(femaleSnakes)):
        gewichteFemaleHidden1 = femaleSnakes[i][0].returnHiddenNeuronsGewichte()
        gewichteFemaleOutput = femaleSnakes[i][0].returnOutputNeuronsGewichte()

        for j in range(len(maleSnakes)):
            if count == len(newSnakes)-1:
                break
            if MUTATIONSART == "ONE-POINT-CROSSOVER":
                for _ in range(math.ceil(GENERATIONSIZE/(len(femaleSnakes)*len(maleSnakes)))):
                    gewichteKombiniertHidden1,gewichteKombiniertOutput = onePointCrossover(femaleSnakes[i][0], maleSnakes[j][0])
                    newSnakes[count] = initNN2(newSnakes[count],INPUTLAYERNEURONS,gewichteKombiniertHidden1,gewichteKombiniertOutput)
                    count = count + 1
            if MUTATIONSART == "MÜNZWURF":
                for _ in range(math.ceil(GENERATIONSIZE/(len(femaleSnakes)*len(maleSnakes)))):
                    #mehrmals sodass die neue Generation so groß ist wie GENERATIONSIZE
                    gewichteMaleHidden1 = maleSnakes[j][0].returnHiddenNeuronsGewichte()
                    gewichteMaleOutput = maleSnakes[j][0].returnOutputNeuronsGewichte()

                    #kombinieren der Hidden-Gewichte des 1.Layers durch durchschnitt und mutation
                    gewichteKombiniertHidden1=[]
                    for u in range(len(gewichteFemaleHidden1)):
                        gewichteKombiniertHidden1.append([kombinieren3(gewichteFemaleHidden1[u][k],gewichteMaleHidden1[u][k]) for k in range(len(gewichteFemaleHidden1[u]))])
                    
                    #kombinieren der Output-Gewichte durch durchschnitt und mutation
                    gewichteKombiniertOutput=[]
                    for u in range(len(gewichteFemaleOutput)):
                        gewichteKombiniertOutput.append([kombinieren3(gewichteFemaleOutput[u][k],gewichteMaleOutput[u][k]) for k in range(len(gewichteFemaleOutput[u]))])
                    
                    if count == len(newSnakes)-1:
                        break
                    newSnakes[count] = initNN2(newSnakes[count],INPUTLAYERNEURONS,gewichteKombiniertHidden1,gewichteKombiniertOutput)
                    count = count + 1
    newSnakes = newSnakes[:GENERATIONSIZE]
    newSnakes = mutieren(newSnakes,zeitStempel)
    return newSnakes
    

def fitness(snake,segments,aktuellerTimestep):
    fitnessWert = len(segments)**2+aktuellerTimestep/TIMESTEPS
    return (snake,fitnessWert)

def fitness2(snake,segments):
    return (snake,len(segments))


# speichert eine Generation in eine pickle file
def saveGeneration(snakes):
    now = datetime.now()
    filename = "snakes_" + IDENTIFIER + "_" + str(round(SIMULATIONSZEIT/3600,2)) + "h_" + str(now.strftime("%H-%M-%S")) + ".pkl"
    with open(PATH + filename, 'wb') as file: 
        pickle.dump(snakes, file)


def loadGeneration(filename):
    with open(filename, 'rb') as pickle_file:
        content = pickle.load(pickle_file)
    return content


#macht einen schritt der snake in richtung von snake_dir und return die neuen felder auf denen sich die snake danach befindet
def schrittmacher(snake_player,snake_dir):
    snake_player_neu=[]
    snake_player_neu.append((snake_player[0][0]+snake_dir[0],snake_player[0][1]+snake_dir[1]))
    snake_player_neu = snake_player_neu + snake_player[0:len(snake_player)-1]
    return snake_player_neu


def printFeld(feld):
    #print(f"{bcolors.GREEN}Warning: No active frommets remain. Continue?{bcolors.ENDC}")
    for i in range(len(feld)):
        for j in range(len(feld[i])):
            if feld[i][j]==1:
                #print(f"{bcolors.GREEN}1{bcolors.ENDC}", end = " ")
                print('#',end=" ")
            elif feld[i][j]==5:
                #print(f"{bcolors.RED}5{bcolors.ENDC}", end = " ")
                #print(Fore.RED + '5')
                print('A',end=" ")
            elif feld[i][j]==8:
                print(f"{bcolors.BLUE}8{bcolors.ENDC}", end = " ")
            else:
                print(".", end = " ")
        print(" ")

def updateFeld(apfel,snake):
    print("-----")
    print(snake)
    Feld_neu = [[0 for _ in range(FELDGROESSE+1)] for _ in range(FELDGROESSE+1)]
    Feld_neu[apfel[0]][apfel[1]] = 5
    for stueck in snake:
        Feld_neu[stueck[0]][stueck[1]] = 1
    return Feld_neu


#erstellen einer neuen generation von snakes
snakes = []
[snakes.append(SnakeBrainNeu()) for _ in range(GENERATIONSIZE)]

for i in range(len(snakes)):
    snakes[i]=initNN(snakes[i],INPUTLAYERNEURONS,HIDDENLAYERNEURONS1,OUTPUTLAYERNEURONS)
#[snake.printNN() for snake in snakes]

avgLength = []
maxLength = []
timeAll = []

alleLengths = [[] for _ in range(NUMBEROFGENERATIONS+1)]

RICHTUNGEN = ["HOCH", "RUNTER", "LINKS", "RECHTS"]

start = time.time()
avgLength = []
maxLength = []
alleLengths = [[] for _ in range(NUMBEROFGENERATIONS+1)]

for gen in range(NUMBEROFGENERATIONS):                      
    #iterieren durch Generationen
    if time.time()-start > SIMULATIONSZEIT:
        break
    generation = []
    print(".", end="")
    for snake in snakes:                                
        #iterieren durch snakes
        #initialisieren der snake erster Eintrag random erste position des kopfes
        snake_player = [(random.randint(int(FELDGROESSE/4),int(3*FELDGROESSE/4)),random.randint(int(FELDGROESSE/4),int(3*FELDGROESSE/4)))]                                                                
        apfel = (random.randint(1,FELDGROESSE),random.randint(1,FELDGROESSE))
        zeitkonto = TIMESTEPS
        alleTimeSteps = 0
        dirs = {'HOCH': 1, 'RUNTER': 0, 'LINKS': 1, 'RECHTS': 1}
        while zeitkonto >= 0:               
            #spielen des Spiels mit einer snake
            zeitkonto = zeitkonto - 1
            alleTimeSteps = alleTimeSteps + 1
            move = getMoveSnake(snake_player[0],snake,apfel,dirs,snake_player,FELDGROESSE)
            if move == "LINKS" or move == "RECHTS" or move == "HOCH" or move == "RUNTER":
                if move == "HOCH" and dirs['HOCH']:
                    snake_dir = (-1, 0)
                    dirs = {'HOCH': 1, 'RUNTER': 0, 'LINKS': 1, 'RECHTS': 1}
                elif move == "RUNTER" and dirs['RUNTER']:
                    snake_dir = (1, 0)
                    dirs = {'HOCH': 0, 'RUNTER': 1, 'LINKS': 1, 'RECHTS': 1}
                elif move == "LINKS" and dirs['LINKS']:
                    snake_dir = (0, -1)
                    dirs = {'HOCH': 1, 'RUNTER': 1, 'LINKS': 1, 'RECHTS': 0}
                elif move == "RECHTS" and dirs['RECHTS']:
                    snake_dir = (0, 1)
                    dirs = {'HOCH': 1, 'RUNTER': 1, 'LINKS': 0, 'RECHTS': 1} 
            snake_player = schrittmacher(snake_player,snake_dir)
            #check ob snake noch auf dem Feld ist
            if snake_player[0][0]>=FELDGROESSE or snake_player[0][0]<0 or snake_player[0][1]>=FELDGROESSE or snake_player[0][1]<0:
                break
            #check ob snake sich selbst frisst
            if snake_player[0] in snake_player[1:len(snake_player)]:
                break
            #apfel gefressen
            if snake_player[0]==apfel:
                snake_player.append((snake_player[len(snake_player)-1][0]-snake_dir[0], snake_player[len(snake_player)-1][1]-snake_dir[1]))
                apfel = random.choice(list(set(Leere_Felder) - set(snake_player)))
                zeitkonto = zeitkonto + TIMESTEPS
        alleLengths[gen].append(len(snake_player))
        generation.append(fitness(snake,snake_player,alleTimeSteps))
    snakes = newGeneration(generation,time.time()-start)
    avgLength.append(sum(alleLengths[gen])/len(alleLengths[gen]))
    timeAll.append(time.time()-start)

saveGeneration(snakes)
print("GESAMTZEIT: " + str(time.time()-start))
filename2 = "alle_Längen_und_Zeiten_" + IDENTIFIER + "_" + str(gen) + "_" + str(round(SIMULATIONSZEIT/3600,2)) + "h_" + str(time.time()-start) + ".pkl"
with open(PATH + filename2, 'wb') as file: 
    pickle.dump((alleLengths,timeAll), file)
print(timeAll)
print(avgLength)
