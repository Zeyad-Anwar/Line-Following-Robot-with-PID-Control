from __future__ import print_function
import struct, sys, argparse, math
import numpy as np

PythonGateways = 'pythonGateways/'
sys.path.append(PythonGateways)
import VsiCommonPythonApi as vsiCommonPythonApi
import VsiTcpUdpPythonGateway as vsiEthernetPythonGateway

PATH_TYPE = "straight"

def make_path():
    if PATH_TYPE == "straight":
        return np.column_stack([np.linspace(0, 10.0, 200), np.zeros(200)])
    
    pts = []

    for x in np.linspace(0, 2, 40): pts.append([x, 0.0])

    for x in np.linspace(2, 6, 80):
        y = 2.0 - 2.0 * math.cos(math.pi * (x - 2.0) / 4.0)
        pts.append([x, y])

    for x in np.linspace(6, 8, 40): pts.append([x, 4.0])

    for x in np.linspace(8, 12, 80):
        y = 2.0 + 2.0 * math.cos(math.pi * (x - 8.0) / 4.0)
        pts.append([x, y])

    for x in np.linspace(12, 14, 40): pts.append([x, 0.0])
    
    return np.array(pts)

class MySignals:
    def __init__(self):
        self.x=self.y=self.theta=0.0
        self.v=self.omega=self.lat_err=self.head_err=0.0

srcMacAddress       = [0x00,0x00,0x00,0x02,0x00,0x01]
SimulatorIpAddress  = [192,168,10,1]
VisualizerIpAddress = [192,168,10,3]
srcIpAddress        = [192,168,10,2]

PORT_SIM_TO_CTRL = 8001
PORT_CTRL_TO_SIM = 8005
PORT_CTRL_TO_VIZ = 8006

Kp=1.5; Ki=0.3; Kd=0.2; ALPHA=0.8; V_REF=1.0; OMEGA_MAX=2.0

def angle_diff(a,b):
    d=a-b
    while d> math.pi: d-=2*math.pi
    while d<-math.pi: d+=2*math.pi
    return d

def nearest_segment(path,x,y):
    pt=np.array([x,y]); min_dist=np.inf; lat_err=head_ref=0.0
    for i in range(len(path)-1):
        A,B=path[i],path[i+1]; AB=B-A; l=np.linalg.norm(AB)
        if l<1e-9: continue
        t=np.clip(np.dot(pt-A,AB)/l**2,0,1)
        dist=np.linalg.norm(pt-(A+t*AB))
        if dist<min_dist:
            min_dist=dist
            lat_err=-(AB[0]*(pt[1]-A[1])-AB[1]*(pt[0]-A[0]))/l
            head_ref=math.atan2(AB[1],AB[0])
    return lat_err,head_ref

class Controller:
    def __init__(self, args):
        self.componentId=1
        self.localHost=args.server_url
        self.domain=args.domain
        self.portNum=50102
        self.simulationStep=self.totalSimulationTime=0
        self.stopRequested=False
        self.hSim = 0   
        self.hViz = 0   
        self.hRecv= 0   
        self.mySignals=MySignals()
        self._integral=self._prev_err=0.0
        self.path=make_path()

    def mainThread(self):
        dSession=vsiCommonPythonApi.connectToServer(
            self.localHost,self.domain,self.portNum,self.componentId)
        vsiEthernetPythonGateway.initialize(
            dSession,self.componentId,bytes(srcMacAddress),bytes(srcIpAddress))
        try:
            vsiCommonPythonApi.waitForReset()
            self.updateInternalVariables()
            if vsiCommonPythonApi.isStopRequested(): raise Exception("stopRequested")

            self.hRecv = vsiEthernetPythonGateway.tcpConnect(
                bytes(SimulatorIpAddress), PORT_SIM_TO_CTRL)
            self.hSim  = vsiEthernetPythonGateway.tcpListen(PORT_CTRL_TO_SIM)
            self.hViz  = vsiEthernetPythonGateway.tcpListen(PORT_CTRL_TO_VIZ)
            print(f"[Controller] Connected  hRecv={self.hRecv} hSim={self.hSim} hViz={self.hViz}")

            nextExpectedTime=vsiCommonPythonApi.getSimulationTimeInNs()
            while vsiCommonPythonApi.getSimulationTimeInNs()<self.totalSimulationTime:
                self.updateInternalVariables()
                if vsiCommonPythonApi.isStopRequested(): raise Exception("stopRequested")
                if vsiEthernetPythonGateway.isTerminationOnGoing(): break
                if vsiEthernetPythonGateway.isTerminated(): break

                rd=vsiEthernetPythonGateway.recvEthernetPacket(PORT_SIM_TO_CTRL)
                if rd[3]!=0: self.decapsulate(rd)

                self.computePID()

                sim_data=(self.pack('d',self.mySignals.v)+self.pack('d',self.mySignals.omega))
                viz_data=(sim_data+self.pack('d',self.mySignals.lat_err)
                         +self.pack('d',self.mySignals.head_err))
                vsiEthernetPythonGateway.sendEthernetPacket(self.hSim, bytes(sim_data))
                vsiEthernetPythonGateway.sendEthernetPacket(self.hViz, bytes(viz_data))

                print(f"\n+=Controller+=  t={vsiCommonPythonApi.getSimulationTimeInNs()} ns")
                print(f"  in : x={self.mySignals.x:.3f}  y={self.mySignals.y:.3f}  theta={self.mySignals.theta:.3f}")
                print(f"  out: v={self.mySignals.v:.3f}  omega={self.mySignals.omega:.3f}  lat_err={self.mySignals.lat_err:.3f}")

                self.updateInternalVariables()
                if vsiCommonPythonApi.isStopRequested(): raise Exception("stopRequested")
                nextExpectedTime+=self.simulationStep
                if vsiCommonPythonApi.getSimulationTimeInNs()>=nextExpectedTime: continue
                if nextExpectedTime>self.totalSimulationTime:
                    vsiCommonPythonApi.advanceSimulation(self.totalSimulationTime-vsiCommonPythonApi.getSimulationTimeInNs()); break
                vsiCommonPythonApi.advanceSimulation(nextExpectedTime-vsiCommonPythonApi.getSimulationTimeInNs())

            if vsiCommonPythonApi.getSimulationTimeInNs()<self.totalSimulationTime:
                vsiEthernetPythonGateway.terminate()
        except Exception as e:
            if str(e)=="stopRequested": vsiCommonPythonApi.advanceSimulation(self.simulationStep+1)
            else: print(f"Controller error: {e}")
        except: vsiCommonPythonApi.advanceSimulation(self.simulationStep+1)

    def computePID(self):
        dt=max(self.simulationStep/1e9,1e-6)
        lat_err,head_ref=nearest_segment(self.path,self.mySignals.x,self.mySignals.y)
        head_err=angle_diff(head_ref,self.mySignals.theta)
        e=lat_err+ALPHA*head_err
        self._integral+=e*dt
        deriv=(e-self._prev_err)/dt; self._prev_err=e
        self.mySignals.omega=float(np.clip(Kp*e+Ki*self._integral+Kd*deriv,-OMEGA_MAX,OMEGA_MAX))
        self.mySignals.v=float(V_REF*max(0.2,1.0-0.5*abs(lat_err)))
        self.mySignals.lat_err=lat_err; self.mySignals.head_err=head_err

    def decapsulate(self,rd):
        p=bytes(rd[2][:rd[3]])
        self.mySignals.x,p=self.unpack('d',p)
        self.mySignals.y,p=self.unpack('d',p)
        self.mySignals.theta,p=self.unpack('d',p)

    def pack(self,t,v): return struct.pack(f'={t}',v)
    def unpack(self,t,b):
        n={'d':8,'f':4,'i':4,'q':8,'h':2,'b':1,'B':1,'?':1}[t]
        return struct.unpack(f'={t}',b[:n])[0],b[n:]
    def updateInternalVariables(self):
        self.totalSimulationTime=vsiCommonPythonApi.getTotalSimulationTime()
        self.stopRequested=vsiCommonPythonApi.isStopRequested()
        self.simulationStep=vsiCommonPythonApi.getSimulationStep()

def main():
    p=argparse.ArgumentParser(" ")
    p.add_argument('--domain',metavar='D',default='AF_UNIX')
    p.add_argument('--server-url',metavar='CO',default='localhost')
    Controller(p.parse_args()).mainThread()

if __name__=='__main__': main()
