import cv2
import numpy as np
from cv2.ximgproc import guidedFilter
from scipy import ndimage
import scipy as sp
import heapq
global w,bi,gi,ri,img
 
def dark_channel(i):
    m,n,_=i.shape
    #padding an array with i
    padded=np.pad(i, ((int(w/2),int(w/2)),(int(w/2),int(w/2)),(0,0)),'edge')
    #creating the dark channel array
    dark=np.zeros((m,n))
    #select the lowest pixels
    for i,j in np.ndindex(dark.shape):
        dark[i,j]=np.min(padded[i:i+w, j:j+w , : ])
    return dark

def bright_channel(i):
    m,n,_ = i.shape
    #padding an array with i
    padded=np.pad(i,((int(w/2),int(w/2)),(int(w/2),int(w/2)),(0,0)),'edge')
    #creating the bright channel array
    bright=np.zeros((m,n))
    #select the highest pixels
    for i,j in np.ndindex(bright.shape):
        bright[i,j]=np.max(padded[i:i+w, j:j+w , : ])
    return bright

def channel_intensities(img):
    b,g,r= cv2.split(img)
    t=img.size/3
    bx=float(np.sum(b))/t
    gx=float(np.sum(g))/t
    rx=float(np.sum(r))/t
    var={bx:bi,gx:gi,rx:ri}
    cmax=var.get(max(var))   
    cmin=var.get(min(var))
    if((cmax==1 or cmax==2) and (cmin==1 or cmin==2)):
        cmid=0
    if((cmax==0 or cmax==2) and (cmin==0 or cmin==2)):
        cmid=1
    if((cmax==0 or cmax==1) and (cmin==0 or cmin==1)):
        cmid=2
    return cmax,cmid,cmin,bx,gx,rx,b,g,r


def bgsubr(i,bright):
    m,n = bright.shape
    #getting indexes of max, min med color channels
    cmax, cmid,cmin,_,_,_,_,_,_=channel_intensities(img)
    bgsubr=np.zeros((m,n))
    arrcmax=i[...,cmax]
    arrcmid=i[...,cmid]
    arrcmin=i[...,cmin]
    for mi in range(m):
        for ni in range(n):
            bgsubr[mi][ni]=1-max(max(arrcmax[mi][ni]-arrcmin[mi][ni],0),max(arrcmid[mi][ni]-arrcmin[mi][ni],0))
    return bgsubr

def rectify_bright(bgsubr):
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lambd = (hsv[...,1].max())/255
    ibright = (bright * lambd) + (bgsubr*(1-lambd))
    return ibright
 
def atmosphoric_light(i,ibright):
    m,n=ibright.shape
    at= np.empty(3)
    selectvar= []

    flati = i.reshape(m*n,3)
    
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    
    win_var=ndimage.generic_filter(gray,np.var,size=3)
    minvar=256
    
    flatbright=ibright.ravel()
    top=heapq.nsmallest(int(m*n*0.1),flatbright)
    
    a=np.where(np.isin(ibright,top))
    for n in range(0,len(a[0])):
         (b,c)=(a[0][n],a[1][n])
         selectvar.append(win_var[b,c])
         if (minvar>np.amin(selectvar)):
             minvar = np.amin(selectvar)
             ib, ic = b,c
             if(minvar == 0): break
    at[0]=i[ib,ic,0]
    at[1]=i[ib,ic,1]
    at[2]=i[ib,ic,2]
    return at
         

def initial_transimission(a,ibright):
    m,n=ibright.shape
    init = np.zeros((m,n))
    
    for i in range(3):
        init = init + ( (ibright-a[i]) / (1.-a[i]) )
        init= (init- np.min(init))/(np.max(init)-np.min(init))
    init = init/3
    return (init- np.min(init))/(np.max(init)- np.min(init))


def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r,r))
    m_p = cv2.boxFilter(p, -1, (r,r))
    m_Ip = cv2.boxFilter(I*p, -1, (r,r))
    cov_Ip = m_Ip-m_I*m_p

    m_II = cv2.boxFilter(I*I, -1, (r,r))
    var_I = m_II-m_I*m_I

    a = cov_Ip/(var_I+eps)
    b = m_p-a*m_I

    m_a = cv2.boxFilter(a, -1, (r,r))
    m_b = cv2.boxFilter(b, -1, (r,r))
    return m_a*I+m_b



def refined_transmission(init):
    refined = np.full_like(init,0)
    gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    eps = 0.2 * 0.2
    eps =eps * 255 * 255
    refined = guidedfilter(gray, init,81,0.01)
    return refined

def restoration_image(i,a,refined):
    m,n,_=i.shape
    corrected=np.broadcast_to(refined[:,:,None],(refined.shape[0],refined.shape[1],3))
    j= ((i-a)/corrected)+a
    return j

def histogram_equalization(j):
  
    M, N,_ = j.shape #Dimensions of the restored image
    #Creating the means of the color channels
    bluemean, greenmean, redmean = float("%0.5f" % (2)), float("%0.5f" % (2)),float("%0.5f" % (2))
    #Handling the case of negative pixels
    for mi in range (M):
        for ni in range (N):
            if (j[mi,ni,0] <= 0):
                j[mi, ni,0] = 0
            if (j[mi,ni,1] <= 0):
                j[mi,ni,1] = 0
            if (j[mi,ni,2] <= 0):
                j[mi,ni, 2] = 0
    #Getting the means and arrays of each channel
    _,_,_,b,g,r, barr,garr,rarr = channel_intensities (j*255)
    #Converting the intensity range to [0,1]
    barr=barr/255
    garr=garr/255
    rarr=rarr/255
    gidx=0.5
    bidx=0.5
    ridx=0.5
 #Equalizing the blue channel
    if (bidx>0):
        bint = float("%0.5f" % (bidx))
        while bluemean != bint:
            bluemean=float("%0.5f" % (float((np.sum(barr))) / (M*N)))
            powb = np.log(bint)/np.log(bluemean)
            barr = (barr)**(powb)
            
  #Equalizing the green channel
    if (gidx>0):
        gint = float("%0.5f" % (gidx))
        while greenmean != gint:
            greenmean =float("%0.5f" % (float((np.sum (garr))) / (M*N)))
            powg = np.log(gint)/np.log(greenmean)
            garr = (garr)**(powg)
    #Equalizing the red channel
    if (ridx>0):
        rint = float("%0.5f" % (ridx))
        while redmean != rint:
            redmean=float("%0.5f" % (float((np.sum(rarr))) / (M*N)))
            powr = np.log(rint)/ np.log(redmean)
            rarr = (rarr) ** (powr)
    #Combining the three channels into the new restored image
    for mi in range (M):
        for ni in range (N):
            j[mi, ni,0]=barr[mi, ni]
            j[mi,ni,1]=garr[mi, ni]
            j[mi,ni, 2]=rarr[mi, ni]
    return j
#####----------------main-----------------
w=15 #window size
bi,gi,ri=0,1,2  #renk kanalları index
print("Resim okunuyor..");

# resmi 3 kanallı numpy dizine çevirmek
img=cv2.imread("3.png")
i=np.asarray(img,dtype=np.float64) 
i=i[:,:,:3]/255

height,width,_ = i.shape
print('Resim boyutu: (Boy:',height, ', En: ',width, ')')
if (height > 600 and width >600):
    print("Resim boyutu cok buyuk")
print('\n')
cv2.imwrite("original_image.jpeg",img)
print("Process bright channel...")
bright=bright_channel(i)
cv2.imwrite("brigh_channel.jpeg",bright*255)
print("bright channel saved successfully")
print("\n")

print("procession the bright channel rectification...")
bgsubr=bgsubr(i,bright)
ibright=rectify_bright(bgsubr)
cv2.imwrite("rectified_bright.jpeg",ibright*255)
print("Rectified bright channel saved successfully")
print("\n")


print ("processing atmospheric light...")
a= atmosphoric_light(i, ibright)
print ("atmospheric light: {}".format(a))

#initial transmission
print("processing initial transmission..")
init = initial_transimission(a, ibright)
white= np.full_like(bright,255)
cv2.imwrite("initial_transmission.jpeg", init*white)
print("initial_transmission saved successfully")
print("\n")

#refined transmission
print("Processing refined transmission...")
refined=refined_transmission(init)
cv2.imwrite("refined_trans.jpeg",refined*white)
print("refined_transmission saved successfully")
print("\n")

#restored image
print("Processing image restoration...")
j=restoration_image(i, a, refined)
cv2.imwrite("resored_img.jpeg",j*255)
print("Restored img saved successfully")
print("\n")

# histogram
print("Processing histogram equalizatition...")
result= histogram_equalization(j)
cv2.imwrite("Final_result.jpeg",result*255)
print("Result saved successfullyy")




