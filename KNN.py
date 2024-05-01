import math
import random

AKTIVIERUNGSFUNKTIONHIDDEN = "Rectifier"
AKTIVIERUNGSFUNKTIONOUTPUT = "Gleich"


class Neuron:
    gewichte = []
    layer = 0
    nummer = 0
    #https://de.wikipedia.org/wiki/K%C3%BCnstliches_Neuron#Aktivierungsfunktionen
    aktivierungsFunktion = "Rectifier"

    
    def __init__(self,gewichte,aktivierungsFunktion,layer,nummer):
        self.gewichte = gewichte
        self.aktivierungsFunktion = aktivierungsFunktion
        self.layer = layer
        self.nummer = nummer


    def outputBerechnen(self,inputs):
        if(len(inputs)!=len(self.gewichte)):
            print("ERROR: len(inputs)!=len(self.gewichte)")
        #1. Berechnen der gewichteten Summe
        gewichteteSumme = sum([inputs[i]*self.gewichte[i] for i in range(len(inputs))])

        #2. Anweden der Aktivierungsfunktion
        #https://de.wikipedia.org/wiki/Rectifier_(neuronale_Netzwerke)
        if self.aktivierungsFunktion == "Rectifier":
            return max(0,gewichteteSumme)
        #https://de.wikipedia.org/wiki/Heaviside-Funktion
        elif self.aktivierungsFunktion == "Heaviside":
            if gewichteteSumme < 0:
                return 0
            else:
                return 1
        #https://de.wikipedia.org/wiki/K%C3%BCnstliches_Neuron#/media/Datei:Piecewise-linear-function.svg
        elif self.aktivierungsFunktion == "Linear":
            if gewichteteSumme > 0.5:
                return 1
            elif gewichteteSumme > -0.5 and gewichteteSumme <= 0.5:
                return gewichteteSumme + 0.5
            elif gewichteteSumme <= -0.5:
                return 0
        #https://de.wikipedia.org/wiki/Sigmoidfunktion
        elif self.aktivierungsFunktion == "Sigmoid":
            return 1/(1+math.exp(-gewichteteSumme))
        #https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
        elif self.aktivierungsFunktion == "Tanh":
            return (math.exp(gewichteteSumme)-math.exp(-gewichteteSumme))/(math.exp(gewichteteSumme)+math.exp(-gewichteteSumme))
        #unveränderter Werte wird returnt
        elif self.aktivierungsFunktion == "Gleich":
            return gewichteteSumme
        
    def printNeuron(self):
        print("LAYER: " + str(self.layer), end = " ")
        print("NUMMER: " + str(self.nummer), end = " ")
        print("GEWICHTE: " + str(self.gewichte))    


class SnakeBrainNeu:
    inputNeurons = []
    hiddenNeurons = []
    outputNeurons = []

    def outputBerechnenNN(self,inputs):
        # input wird an hidden layer weitergeleitet
        outputHiddenNeurons = []
        for i in range(len(self.hiddenNeurons)):
            outputHiddenNeurons.append(self.hiddenNeurons[i].outputBerechnen(inputs))
        # ouput des hidden layer wird an output layer weitergeleitet
        outputOutputNeurons = []
        for i in range(len(self.outputNeurons)):
            outputOutputNeurons.append(self.outputNeurons[i].outputBerechnen(outputHiddenNeurons))

        return outputOutputNeurons
    
    def printNN(self):
        print("Inputlayer: ")
        [neuron.printNeuron() for neuron in self.inputNeurons]
        print("\n\n")
        print("Hiddenlayer: ")
        [neuron.printNeuron() for neuron in self.hiddenNeurons]
        print("\n\n")
        print("Outputlayer: ")
        [neuron.printNeuron() for neuron in self.outputNeurons]
        print("Hello-----------------------")
    
    def returnHiddenNeuronsGewichte(self):
        gewichte = []
        for i in range(len(self.hiddenNeurons)):
            gewichte.append(self.hiddenNeurons[i].gewichte)

        return gewichte
    
    def returnOutputNeuronsGewichte(self):
        gewichte = []
        for i in range(len(self.outputNeurons)):
            gewichte.append(self.outputNeurons[i].gewichte)

        return gewichte
    

#initNN: gibt ein initialisiertes NN mit input, 1. hidden, 2. hidden und output layer zurück
def initNN(snake,anzahlInputNeurons,anzahlHiddenNeurons, anzahlOutputNeurons):
    inputN = []
    for i in range(anzahlInputNeurons):
        inputN.append(Neuron([1],"Gleich",0,i))
    snake.inputNeurons = inputN

    hiddenN = []
    for i in range(anzahlHiddenNeurons):
        hiddenN.append(Neuron([random.uniform(-1,1) for j in range(anzahlInputNeurons)],AKTIVIERUNGSFUNKTIONHIDDEN,1,i))

    snake.hiddenNeurons = hiddenN
   

    outputN = []
    for i in range(anzahlOutputNeurons):
        outputN.append(Neuron([random.uniform(-1,1) for j in range(anzahlHiddenNeurons)],AKTIVIERUNGSFUNKTIONOUTPUT,2,i))

    snake.outputNeurons = outputN

    return snake


def initNN2(snake, anzahlInputNeurons ,gewichteHidden, gewichteOutput):
    inputN = []
    for i in range(anzahlInputNeurons):
        inputN.append(Neuron([1],"Gleich",0,i))
    snake.inputNeurons = inputN

    hiddenN = []
    for i in range(len(gewichteHidden)):
        hiddenN.append(Neuron(gewichteHidden[i],AKTIVIERUNGSFUNKTIONHIDDEN,1,i))
    snake.hiddenNeurons = hiddenN

    outputN = []
    for i in range(len(gewichteOutput)):
        outputN.append(Neuron(gewichteOutput[i],AKTIVIERUNGSFUNKTIONOUTPUT,2,i))
    snake.outputNeurons = outputN
    
    return snake
