import numpy as np
import cv2

class ImageAreaSelection:
    def __init__(self, inputImage):
        self.mouseClickPoints = [] #A list storing the mouse click points or the coordinates of quadrilateral
        #Name of the window to show image
        self.windowName = f"Image with irregular quadrilateral"
        #The original image
        self.raw_img = inputImage.copy()
        #Make a copy of the original image and show the selected area
        self.imgWidthQuadrilaterals = self.raw_img.copy()
        self.redo_flag = False
    
# =============================================================================
#     def __del__(self):  #destructor
#         print(f'calling __del__()')
#         cv2.destroyWindow(self.windowName)
# =============================================================================
    
    #Callback functions for mouse event
    def mouseClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: #Check if left mouse button is pressed down or not
            refPt = [(x, y)] 
            pt = tuple(refPt[0])
            cv2.circle(self.imgWidthQuadrilaterals,pt, 3, (0,255,255), -1)
            cv2.putText(self.imgWidthQuadrilaterals,'{}'.format(len(self.mouseClickPoints)+1),pt, cv2.FONT_HERSHEY_SIMPLEX, 2, (200,255,155),5) #---write the text
            self.mouseClickPoints.append(refPt[0])
            cv2.imshow(self.windowName, self.imgWidthQuadrilaterals)
            print(refPt)
        
    def letUserClickPoints(self):
        cv2.namedWindow( self.windowName  ,cv2.WINDOW_NORMAL)
        cv2.setMouseCallback( self.windowName, self.mouseClick)
        cv2.putText(self.imgWidthQuadrilaterals,self.windowName,(0,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (200,255,155),5) #---write the text
        
        cv2.imshow(self.windowName, self.imgWidthQuadrilaterals)
        #Wait indefinitly for user mouse clicks until pressing any key on keyboard
        print(f'Click your mouse to select area. Press any key to continue....')
        cv2.waitKey(0)
        cv2.destroyWindow(self.windowName)

    #Draw quadrilateral on the image
        mouseClickPoint_length = len( self.mouseClickPoints )

        for i in range(mouseClickPoint_length):
            cv2.polylines(self.imgWidthQuadrilaterals, np.array([self.mouseClickPoints]),True, (0,0,255),5)
        
    def ShowSelectedArea(self):
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(self.windowName, self.imgWidthQuadrilaterals)
        print("Are you sure? [Y/N]")
# =============================================================================
#         while True:
#             if cv2.waitKey(1) & 0xFF == ord('y'):
#                 break
#             elif cv2.waitKey(1) & 0xFF == ord('n'):
#                 self.redo_flag = True
#                 break
# =============================================================================
        cv2.waitKey(0)
        cv2.destroyWindow(self.windowName)
    
# ********************************************
# End of class ImageAreaSelection
# ********************************************    
# if __name__ == '__main__':
#     #Global variables
#     inputImage = cv2.imread('highway.jpg', cv2.IMREAD_COLOR)
#     myImageAreaSelector = ImageAreaSelection( inputImage )
#     myImageAreaSelector.letUserClickPoints()
#     myImageAreaSelector.ShowSelectedArea()
    
#     #cv2.destroyAllWindows()
    
