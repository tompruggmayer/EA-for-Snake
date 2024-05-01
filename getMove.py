import math


def getMoveSnake(snake,snakeBrain,food,dirs,segments,FELDGROESSE):
    input = []

    # abstand zwischen snake und food
    abstand = math.sqrt(((snake[0]-food[0])/FELDGROESSE)**2+((snake[1]-food[1])/FELDGROESSE)**2)

    input.append(abstand)

    if food[0]-snake[0] > 0:
        input.append(1)
    else:
        input.append(0)
    if food[1]-snake[1] > 0:
        input.append(1)
    else:
        input.append(0)


    if (snake[0]-1,snake[1]-1) in segments[1:len(segments)] or snake[0]-1<0 or snake[1]-1<0:      #Feld oben links
        input.append(1)
    else:
        input.append(0)
    if (snake[0]-1,snake[1]) in segments[1:len(segments)] or snake[0]-1<0:          #Feld oben mitte
        input.append(1)
    else:
        input.append(0)
    if (snake[0]-1,snake[1]+1) in segments[1:len(segments)] or snake[0]-1<0 or snake[1]+1>FELDGROESSE:      #Feld oben rechts
        input.append(1)
    else:
        input.append(0)
    if (snake[0],snake[1]-1) in segments[1:len(segments)] or snake[1]-1<0:          #Feld links
        input.append(1)
    else:
        input.append(0)
    if (snake[0],snake[1]+1) in segments[1:len(segments)] or snake[1]+1>FELDGROESSE:      #Feld rechts
        input.append(1)
    else:
        input.append(0)
    if (snake[0]+1,snake[1]-1) in segments[1:len(segments)] or snake[0]+1>FELDGROESSE or snake[1]-1<0:          #Feld unten links
        input.append(1)
    else:
        input.append(0)
    if (snake[0]+1,snake[1]) in segments[1:len(segments)] or snake[0]+1>FELDGROESSE:      #Feld unten mitte
        input.append(1)
    else:
        input.append(0)
    if (snake[0]+1,snake[1]+1) in segments[1:len(segments)] or snake[0]+1>FELDGROESSE or snake[1]+1>FELDGROESSE:      #Feld unten rechts
        input.append(1)
    else:
        input.append(0)


    #RICHTUNG DER SNAKE
    input.append(dirs['HOCH'])
    input.append(dirs['RUNTER'])
    input.append(dirs['LINKS'])
    input.append(dirs['RECHTS'])

    input.append(len(segments)/(FELDGROESSE**2))

    output = snakeBrain.outputBerechnenNN(input)

    if output.index(max(output)) == 0:
        #print("LINKS")
        return "LINKS"
    elif output.index(max(output)) == 1:
        #print("RECHTS")
        return "RECHTS"
    elif output.index(max(output)) == 2:
        #print("UP")
        return "HOCH"
    elif output.index(max(output)) == 3:
        #print("RUNTER")
        return "RUNTER"
    else:
        print("ERROR: Zu viele Output-Neuronen?")
    