import cv2
import numpy as np
import SelectImageArea as sa

if __name__ == '__main__':
    #Global variables
    inputImage = cv2.imread('highway.jpg', cv2.IMREAD_COLOR)
    myImageAreaSelector = sa.ImageAreaSelection( inputImage )
    myImageAreaSelector.letUserClickPoints()
    myImageAreaSelector.ShowSelectedArea()
    # More example
    
    cv2.namedWindow( f'Result' ,cv2.WINDOW_NORMAL)
    imageWithQuadliterals = inputImage.copy()
    #Draw quadliterals on the image
    Point_length = len( myImageAreaSelector.mouseClickPoints )
    #Make Point_length multiple of 4
    Point_length =( Point_length >> 2 ) << 2

    for i in range( 0, Point_length, 4 ):
        #cv2.polylines(imageWithQuadliterals, np.array([myImageAreaSelector.mouseClickPoints[i:i+4]]),True, (0,200,0),3)
        points = np.array( [ myImageAreaSelector.mouseClickPoints[i:i+4] ], dtype=np.int32 )
        cv2.fillPoly( imageWithQuadliterals, [points], (0, 200, 0, 0.5) )
             
    cv2.imshow( f'Result', imageWithQuadliterals )
    cv2.waitKey(0)
    cv2.destroyWindow( f'Result' )
    #cv2.destroyAllWindows()