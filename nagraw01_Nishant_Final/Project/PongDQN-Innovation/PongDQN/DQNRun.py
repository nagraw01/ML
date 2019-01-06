
#  https://github.com/llSourcell/pong_neural_network_live

import MyPong # My PyGame Pong Game 
import MyAgent # My DQN Based Agent
import numpy as np 
import random 
import matplotlib.pyplot as plt

#   DQN Algorith Paramaters 
ACTIONS = 3 # Number of Actions.  Acton istelf is a scalar:  0:stay, 1:Up, 2:Down
STATECOUNT = 5 # Size of State [ PlayerYPos, BallXPos, BallYPos, BallXDirection, BallYDirection] 
TOTAL_GAMETIME = 15000
# =======================================================================
# Normalise GameState
def CaptureNormalisedState(PlayerYPos, BallXPos, BallYPos, BallXDirection, BallYDirection):
	gstate = np.zeros([STATECOUNT])
	gstate[0] = PlayerYPos/400.0	# Normalised PlayerYPos
	gstate[1] = BallXPos/400.0	# Normalised BallXPos
	gstate[2] = BallYPos/400.0	# Normalised BallYPos
	gstate[3] = BallXDirection/1.0	# Normalised BallXDirection
	gstate[4] = BallYDirection/1.0	# Normalised BallYDirection
	
	return gstate
# =====================================================================
# Main Experiment Method 
def PongDQN():
	GameTime = 0
    
	GameHistory = []
	
	#Create our PongGame instance
	TheGame = MyPong.PongGame()
    # Initialise Game
	TheGame.InitialDisplay()

	TheAgent = MyAgent.Agent(STATECOUNT, ACTIONS)
	
	
	BestAction = 0
	
	GameState = CaptureNormalisedState(200.0, 200.0, 200.0, 1.0, 1.0)
	
    
	for gtime in range(TOTAL_GAMETIME):    
	
		
		if GameTime % 100 == 0:
			TheGame.UpdateGameDisplay(GameTime,TheAgent.epsilon)

		
		BestAction = TheAgent.Act(GameState)
		
		
		[ReturnScore,PlayerYPos, BallXPos, BallYPos, BallXDirection, BallYDirection]= TheGame.PlayNextMove(BestAction)
		NextState = CaptureNormalisedState(PlayerYPos, BallXPos, BallYPos, BallXDirection, BallYDirection)
		
		
		TheAgent.CaptureSample((GameState,BestAction,ReturnScore,NextState))
		
		
		TheAgent.Process()
		
	
		GameState = NextState
		
		
		GameTime = GameTime+1

      
		if GameTime % 1000 == 0:
          
			donothing =0

		if GameTime % 200 == 0:
			print("Timestep: ", GameTime," Score: ", "{0:.2f}".format(TheGame.GScore), "   EPSILON: ", "{0:.4f}".format(TheAgent.epsilon))
			GameHistory.append((GameTime,TheGame.GScore,TheAgent.epsilon))
			

	x_val = [x[0] for x in GameHistory]
	y_val = [x[1] for x in GameHistory]

	plt.plot(x_val,y_val)
	plt.xlabel("Game Time")
	plt.ylabel("Score")
	plt.show()

	

def main():
    
	
	PongDQN()
	
	
if __name__ == "__main__":
    main()
