
import pyautogui

screenWidth, screenHeight = pyautogui.size() # Get the size of the primary monitor.
currentMouseX, currentMouseY = pyautogui.position() # Get the XY position of the mouse.
print('screenWidth={0}, screenHeight={1}, currentMouseX={2}, currentMouseY={3}'.format(screenWidth, screenHeight, currentMouseX, currentMouseY))
#pyautogui.moveRel(x축 참조값, y축 참조값, 시간)
pyautogui.moveRel(100, 100)
pyautogui.moveRel(0, 500, 5)

pyautogui.moveTo(screenWidth/2, screenHeight/2)
pyautogui.moveTo(currentMouseX, currentMouseY, 5)
#pyautogui.moveTo(200, 400)
#pyautogui.moveTo(300, 200, 2)
#pyautogui.moveRel(0, 200, 2)
#pyautogui.moveRel(200, 0, 2)

