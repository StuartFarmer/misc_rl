import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv
import talib

underlying_data = []

# open the data
with open('./3m_data.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		underlying_data.append(row)

# scrape only the closing prices
underlying_data = [float(d[4]) for d in underlying_data[1:]]
underlying_data = np.array(underlying_data)

# set up the bollinger bands
BBAND_TIMEPERIOD = 50
upperband, middleband, lowerband = talib.BBANDS(underlying_data, timeperiod=BBAND_TIMEPERIOD, nbdevup=1, nbdevdn=1)

# set up the williams alligator indicators
jaw = talib.EMA(underlying_data, timeperiod=13) # 8
lips = talib.EMA(underlying_data, timeperiod=5) # 3

JAW_OFFSET = 8
for i in range(JAW_OFFSET):
	jaw = np.insert(jaw, 0, np.nan)
jaw = jaw[JAW_OFFSET:]

LIPS_OFFSET = 3
for i in range(LIPS_OFFSET):
	lips = np.insert(lips, 0, np.nan)
lips = lips[LIPS_OFFSET:]

# add the instrument data with indicators to a single numpy array
data = np.array([underlying_data[BBAND_TIMEPERIOD:], \
	upperband[BBAND_TIMEPERIOD:], \
	lowerband[BBAND_TIMEPERIOD:], \
	jaw[BBAND_TIMEPERIOD:], \
	lips[BBAND_TIMEPERIOD:]])

# set how many steps the window shows
WINDOW_SIZE = 300

# create the matplotlib window and plots
fig, ax = plt.subplots()

x = np.arange(0, WINDOW_SIZE)

# one line for every subarray in the data array
# you can add as many arrays as you want into the data array as long as they are all the same size
# new subarrays will automatically animate
lines = []
for i in range(len(data)):
	line, = ax.plot(x, data[i][:WINDOW_SIZE])
	lines.append(line,)

# updating the lines abstracted into a callable function for the animation object
def animate(i):
	for ii in range(len(data)):
		lines[ii].set_ydata(data[ii][i:WINDOW_SIZE+i])
	ax.relim()
	ax.autoscale_view(True,True,True)
	return lines

# initializing the plot
def init():
	for line in lines:
		line.set_ydata(np.ma.array(x, mask=True))
	return lines

# initializing the animation object
ani = animation.FuncAnimation(fig, animate, np.arange(1, len(data[0])), init_func=init,
                              interval=25, blit=True)

# its plot time
plt.show()