from paraview.simple import *
canex2=OpenDataFile('C:\\Users\\janve\\local\\masterarbeit\\fenics\\test\\ellipse\\solution.pvd')
clip=Clip()
Hide(canex2)
Show(clip)
ResetCamera()
Render()
SaveScreenshot('C:\\Users\\janve\\local\\masterarbeit\\fenics\\test\\ellipse\\picture.jpg')