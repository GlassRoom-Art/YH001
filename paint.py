import numpy as np
import cv2
import random
import time
import json
import gradient
import os

import math

from rotate_brush import *
from color_table import *
from Differ import *


def rn():
    return random.random()

def limit(x,minimum,maximum):
    return min(max(x,minimum),maximum)

class paint():
    def __init__(self):
        self.m_Brush = Brushes()
        self.m_Brush.load_brushes()
        self.m_Colors = Colors()
        self.m_Colors.color_10()
    
    def load(self,filename='face3.png'):
        print('loading',filename,'...')

        self.imname = filename.split('.')[0]

        # original image
        self.img = cv2.imread(filename)

        xshape = self.img.shape[1]
        yshape = self.img.shape[0]

        self.rescale = xshape/640
        # display rescaling: you'll know when it's larger than your screen
        if self.rescale<1:
            self.rescale=1

        self.xs_small = int(xshape/self.rescale)
        self.ys_small = int(yshape/self.rescale)
        self.smallerimg = cv2.resize(self.img,dsize=(self.xs_small,self.ys_small)).astype('float32')/255
        # for preview purpose,
        # if image too large

        # convert to float32
        self.img = self.img.astype('float32')/255

        # canvas initialized
        self.canvas = self.img.copy()
        self.canvas[:,:] = 0.8
        self.index = np.zeros(self.img.shape[0:2]).astype(int)
        self.stroke_index = 0

        #clear hist
        self.hist=[]
        print(filename,'loaded.')

        self.m_Differ = Differ(xshape,yshape)
    
    def record(self,sth):
        self.hist.append(sth)

    def savehist(self,filename='hist.json'):
        f = open(filename,'w')
        json.dump(self.hist,f)
        f.close()

    def loadhist(self,filename='hist.json'):
        f = open(filename,'r')
        self.hist=[]
        self.hist = json.load(f)
    
    def showimg(self):
        if self.rescale==1:
            smallercanvas = self.canvas
        else:
            smallercanvas = cv2.resize(self.canvas,dsize=(self.xs_small,self.ys_small),interpolation=cv2.INTER_NEAREST)

        i,j,d = self.m_Differ.wherediff(smallercanvas,self.smallerimg)
        sd = np.mean(d)
        print('mean diff:',sd)

        ###francis##
        #d[i,:]=1.0
        #d[:,j]=1.0

        cv2.imshow('canvas',smallercanvas)
        cv2.imshow('img',self.smallerimg)
        cv2.imshow('diff',d)

        cv2.waitKey(10)

    
    def paint_one(self,x,y,brushname='random',angle=-1.,minrad=10,maxrad=60,round=1):
        
        oradius = math.exp(-(round+1)**2)*rn()*rn()*maxrad+minrad
        fatness = 1/(1+rn()*rn()*6)
        delta = 1e-4

        #obtain integer radius and shorter-radius
        def intrad(orad):
            radius = int(orad)
            srad = int(orad*fatness+1)
            return radius,srad

        # get copy of square ROI area, to do drawing and calculate error.
        def get_roi(newx,newy,newrad):
            radius,srad = intrad(newrad)

            xshape = self.img.shape[1]
            yshape = self.img.shape[0]

            yp = int(min(newy+radius,yshape-1))
            ym = int(max(0,newy-radius))
            xp = int(min(newx+radius,xshape-1))
            xm = int(max(0,newx-radius))

            if yp<=ym or xp<=xm:
                # if zero w or h
                raise NameError('zero roi')

            ref = self.img[ym:yp,xm:xp]
            bef = self.canvas[ym:yp,xm:xp]
            aftr = np.array(bef)
            return ref,bef,aftr

        # paint one stroke with given config and return the error.
        def paint_aftr_w(color,angle,nx,ny,nr):
            ref,bef,aftr = get_roi(nx,ny,nr)
            radius,srad = intrad(nr)
            self.m_Brush.compose(aftr,brush,x=radius,y=radius,rad=radius,srad=srad,angle=angle,color=color,N=0,idx=self.index,usefloat=True,useoil=False)
            # if useoil here set to true: 2x slow down + instability

            err_aftr = np.mean(self.m_Differ.new_diff(aftr,ref))
            return err_aftr

        # finally paint the same stroke onto the canvas.
        def paint_final_w(color,angle,nr):
            radius,srad = intrad(nr)
            self.stroke_index += 1
            self.m_Brush.compose(self.canvas,brush,x=x,y=y,rad=radius,srad=srad,angle=angle,color=color,N=self.stroke_index,idx=self.index,usefloat=True,useoil=True)
            # enable oil effects on final paint.
            # np.float64 will cause problems
            rec = [x,y,radius,srad,angle,color[0],color[1],color[2],brushname,self.stroke_index]
            rec = [float(r) if type(r)==np.float64 or type(r)==np.float32 else r for r in rec]
            self.record(rec)
            # log it!

        # given err, calculate gradient of parameters wrt to it
        def calc_gradient(err):
            b,g,r = c[0],c[1],c[2]
            cc = b,g,r
            #print(cc)
            err_aftr = paint_aftr_w((b+delta,g,r),angle,x,y,oradius)
            gb = err_aftr - err

            err_aftr = paint_aftr_w((b,g+delta,r),angle,x,y,oradius)
            gg = err_aftr - err

            err_aftr = paint_aftr_w((b,g,r+delta),angle,x,y,oradius)
            gr = err_aftr - err

            err_aftr = paint_aftr_w(cc,(angle+5.)%360,x,y,oradius)
            ga = err_aftr - err

            err_aftr = paint_aftr_w(cc,angle,x+2,y,oradius)
            gx =  err_aftr - err

            err_aftr = paint_aftr_w(cc,angle,x,y+2,oradius)
            gy =  err_aftr - err

            err_aftr = paint_aftr_w(cc,angle,x,y,oradius+3)
            gradius = err_aftr - err

            return np.array([gb,gg,gr])/delta,ga/5,gx/2,gy/2,gradius/3,err

        
        radius,srad = intrad(oradius)
        brush,key = self.m_Brush.get_brush(brushname)
        #set initial angle
        if angle == -1.:
            angle = rn()*360
        # sample color from image => converges faster.
        c = self.img[int(y),int(x),:]
        c = self.m_Colors.find_nearest_color(c)
        # max and min steps for gradient descent
        tryfor = 10
        mintry = 3

        for i in range(tryfor):
            try: # might have error
                # what is the error at ROI?
                ref,bef,aftr = get_roi(x,y,oradius)
                orig_err = np.mean(self.m_Differ.new_diff(bef,ref))

                # do the painting
                err = paint_aftr_w(c,angle,x,y,oradius)

                #if err * math.exp(1/(round+2)**2) <orig_err  and i > mintry:
                if err*1.1 < orig_err and i > mintry:
                    paint_final_w(c,angle,oradius)
                    return True,i

                # if not satisfactory
                # calculate gradient
                grad,anglegrad,gx,gy,gradius,err = calc_gradient(err)

            except NameError as e:
                print(e)
                print('error within calc_gradient')
                return False,i

            # if printgrad: #debug purpose.
            #     if i==0:
            #         print('----------')
            #         print('orig_err',orig_err)
            #     print('ep:{}, err:{:3f}, color:{}, angle:{:2f}, xy:{:2f},{:2f}, radius:{:2f}'.format(i,err,c,angle,x,y,oradius))

            # do descend
            if i<tryfor-1:
                #print(grad)
                c = c - (grad*.05).clip(max=0.05,min=-0.05)
                c = c.clip(max=1.,min=0.)
                c = self.m_Colors.find_nearest_color(c)
                angle = (angle - limit(anglegrad*100000,-5,5))%360
                x = x - limit(gx*1000*radius,-3,3)
                y = y - limit(gy*1000*radius,-3,3)
                oradius = oradius* (1-limit(gradius*20000,-0.2,.2))
                oradius = limit(oradius,7,100)

                # print('after desc:x:{:2f},y:{:2f},angle:{:2f},oradius:{:5f}'
                # .format(x,y,angle,oradius))

        return False,tryfor
    
    def putstrokes(self,howmany,r):

        def samplepoints():
            # sample a lot of points from one error image - save computation cost

            point_list = []
            y,x,d =self.m_Differ.wherediff(self.canvas,self.img)
            phasemap = gradient.get_phase(self.img)
            #cv2.imshow("phasemap",phasemap)
            # while not enough points:
            while len(point_list)<howmany:
                # randomly pick one point
                yshape, xshape = self.img.shape[0:2]
                ry,rx = int(rn()*yshape),int(rn()*xshape)
                # accept with high probability if error is large
                # and vice versa
                if d[ry,rx]>0.5*rn():
                    # get gradient orientation info from phase map
                    phase = phasemap[ry,rx] # phase should be between [0,2pi)

                    # choose direction perpendicular to gradient
                    angle = (phase/math.pi*180+90)%360
                    # angle = 22.5

                    point_list.append((ry,rx,angle))
            return point_list

        def pcasync(tup):
            y,x,angle = tup
            b,key = self.m_Brush.get_brush(key='random') # get a random brush
            #return paint_one(x,y,brushname=key,minrad=10,maxrad=50,angle=angle) #num of epoch
            return self.paint_one(x, y, brushname=key, minrad=20, maxrad=400, angle=angle,round=r)  # num of epoch

        # if True:
        #     from thready import amap # multithreading
        #     point_list = samplepoints()
        #     return amap(pcasync,point_list)
        #
        # else: # single thre ading test

        point_list = samplepoints()
        res={}
        for idx,item in enumerate(point_list):
            #print('single threaded mode.',idx)
            res[idx] = pcasync(item)
        return res

    def r(self,epoch=1):
        # filename prefix for each run
        seed = int(rn()*1000)
        file_step = open('step.txt','w')
        print('running...')
        st = time.time()

        # timing counter for autosave and showimg()
        timecounter = 0
        showcounter = 0
        totalstep = 0.

        
        for round in range(epoch):
            loopfor = 2
            paranum = 128
            # number of stroke tries per batch, sent to thread pool
            # smaller number decreases efficiency

            succeeded = 0 # how many strokes being placed
            ti = time.time()

            # average step of gradient descent performed
            avgstep=0.
   
            for k in range(loopfor):
                res = self.putstrokes(paranum,round) # res is a map of results

                for r in res:
                    status,step = res[r]
                    avgstep += step
                    succeeded += 1 if status else 0
            
            totalstep += succeeded
            avgstep/=loopfor*paranum

            steptime = time.time()-ti
            tottime = time.time()-st

            #info out
            print('epoch',round,'/',epoch ,'succeeded:',succeeded,'/',loopfor*paranum,'avg step:' ,avgstep,'total step:' ,totalstep,'time:{:.1f}s, total:{:.1f}s'.format(steptime,tottime))

            index_each_round = self.index.reshape(self.index.shape[0] * self.index.shape[1]).astype(int)
            index_each_round = index_each_round.tolist()
            index_set = set(index_each_round)
            index_list=list(index_set)
            print("at epoch %d, index size before is %d"%(round, len(index_list)-1))
            file_step.write(str(round)+"   "+str(len(index_list)-1)+'\n')

            # autosave during canvas painting
            dosaveimage = True
            # dosaveimage = False

            #autosave
            timecounter+=steptime
            if(timecounter>20):
                timecounter=0            
                if dosaveimage:
                    print('saving to disk...')

                    if not os.path.exists('./'+self.imname):
                        os.mkdir('./'+self.imname)

                    cv2.imwrite(self.imname+'/{}_{:04d}.png'.format(seed,round),self.canvas*255)
                    print('saved.')
            # refresh view
            showcounter+=steptime
            if(showcounter>3):
                showcounter=0
                self.showimg()
        self.showimg()
        self.savehist()
        file_step.close()
    def savestroke(self,filename='stroke.json',s_list={}):
        f = open(filename,'w')
        json.dump(s_list,f)
        f.close() 
    def destroy(self):
        cv2.destroyAllWindows()

    
    def repaint(self,stroke_file='stroke.json',constraint_angle=False,upscale=1.,batchsize=16):
        # global index_1,index_list
        starttime = time.time()
        
        self.newcanvas = np.array(self.canvas).astype('uint8')
        # newcanvas = cv2.cvtColor(newcanvas,cv2.COLOR_BGR2BGRA) # fastest format

        # if upscale!=1.:
        #     newcanvas = cv2.resize(newcanvas,dsize=(int(newcanvas.shape[1]*upscale),int(newcanvas.shape[0]*upscale)))

        self.newcanvas[:,:,:] = int(0.8*255)

        def showthis():
            if self.rescale==1:
                smaller_newcanvas = self.canvas
            else:
                smaller_newcanvas = cv2.resize(self.canvas,dsize=(self.xs_small,self.ys_small),interpolation=cv2.INTER_NEAREST)
            #showsize = 640
            #resize_scale = min(showsize/newcanvas.shape[1],1.)
            #resizedx,resizedy = int(newcanvas.shape[1]*resize_scale),int(newcanvas.shape[0]*resize_scale)

            #smallercanvas = cv2.resize(newcanvas,dsize=(resizedx,resizedy),interpolation=cv2.INTER_NEAREST)
            #cv2.imshow('repaint',smallercanvas)
            cv2.imshow('repaint', smaller_newcanvas)


        def paintone(histitem):
            x,y,radius,srad,angle,cb,cg,cr,brushname,p_index = histitem
            #print(x,y,radius,srad,angle,cb,cg,cr,brushname)
            cb,cg,cr = int(cb*255),int(cg*255),int(cr*255)
            # cv2.ellipse(newcanvas,(int(x),int(y)),(radius,srad),angle,0,360,color=(cb,cg,cr),thickness=-1)

            b,key = self.m_Brush.get_brush(brushname)
            #print(angle)
            if constraint_angle:
                angle = constraint_angle+rn()*20-10

            if upscale!=1:
                x,y,radius,srad = x*upscale,y*upscale,radius*upscale,srad*upscale

            # print("strok really painted: ", )

            self.m_Brush.compose(self.newcanvas,b, x=x,y=y,rad=radius,srad=srad,angle=angle,color=[cb,cg,cr],N=p_index,useoil=True)
            return [cb,cg,cr]

        def loadstroke(filename='stroke.json'):
            f_s = open(filename,'r')
            return json.load(f_s)
        # batch = []
        #
        # def runbatch(batch):
        #     from thready import amap # multithreading
        #     return amap(paintone,batch)
        #
        # #lastep = 0
        #
        # i = 0
        # while i < (len(index_list)):
        #     while len(batch)<batchsize and i <len(index_list):
        #         index_to_paint = index_list[i]
        #         if index_to_paint == 0:
        #             i += 1
        #             continue
        #         print("index should paint: ", index_to_paint)
        #
        #         for j in range(len(hist)):
        #             p_stroke = hist[j]
        #             if p_stroke[-1] == index_to_paint:
        #                 print("stroke should paint: ", p_stroke)
        #                 batch.append(p_stroke)
        #                 break
        #         i += 1
        #     runbatch(batch)
        #     batch = []
        index_list=loadstroke()
        i=0
        count=0
        while i <(len(index_list)):
            index_to_paint=index_list[i]
            if index_to_paint == 0:
                i += 1
                continue
            #print("index should paint: ", index_to_paint)

            for j in range(len(self.hist)):
                p_stroke = self.hist[j]
                if p_stroke[-1] == index_to_paint:
                    #print("strok should paint: ", p_stroke)
                    stroke_color=paintone( p_stroke)
                    cv2.imshow('repaint', self.newcanvas)
                    cv2.waitKey(0)
                    #print (stroke_color)
                    if stroke_color ==[255,0,0]:
                        pass
                    else:
                        count+=1 
                    break
            i += 1
        print("Repaint completed.")
        print("step after simplify", count)
        #print(time.time()-starttime,'s elapsed')
        #m,n,diff_repaint = self.m_Differ.wherediff(self.newcanvas/255,self.canvas)
        #mdiff = np.mean(diff_repaint)
        #print('mean diff:',mdiff)
        showthis()
        #cv2.imshow("D",diff_repaint)
        cv2.waitKey(0)
        #return newcanvas
    
    def modify_jsonfile(self):
        result = {}
        result['canve_size_width'] = 640
        result['canve_size_height'] = 640
        result['path'] = []

        def loadstroke(filename='stroke.json'):
            f_s = open(filename,'r')
            return json.load(f_s)

        index_list=loadstroke()
        i=0
        while i <(len(index_list)):
            index_to_paint=index_list[i]
            if index_to_paint == 0:
                i += 1
                continue
        
            for j in range(len(self.hist)):
                p_stroke = self.hist[j]
                if p_stroke[-1] == index_to_paint:
                    i_line = {}
                    x, y, rad, srad, angle, g, b, r, brushname, index = p_stroke
                    i_line['index'] = index
                    i_line['color_index'] = self.m_Colors.colors.index((g,b,r))
                    i_line['pen_index'] = 1
                    i_line['line'] = []
                    x1 = int(x + rad * math.cos((180-angle) * math.pi / 180))
                    y1 = int(y + rad * math.sin((180-angle) * math.pi / 180))
                    x2 = int(x - rad * math.cos((180-angle) * math.pi / 180))
                    y2 = int(y - rad * math.sin((180-angle) * math.pi / 180))
                    i_line['line'].append({'x':x1,'y':y1})
                    i_line['line'].append({'x':x2,'y':y2})
                    result['path'].append(i_line)
            i += 1

            with open('draw.json','w') as f:
                json.dump(result,f, indent=4 , separators = (',',':'), ensure_ascii=False)







    
    
