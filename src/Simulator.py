from __future__ import print_function
import struct, sys, argparse, math
import numpy as np

PythonGateways = 'pythonGateways/'
sys.path.append(PythonGateways)
import VsiCommonPythonApi as vsiCommonPythonApi
import VsiTcpUdpPythonGateway as vsiEthernetPythonGateway

PATH_TYPE    = "straight"
SPAWN_OFFSET = (0.0, 0.5, 0.1)
NOISE_STD    = 0.0
DISTURBANCE  = 0.0

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
        self.v = self.omega = 0.0
        self.x = self.y = self.theta = 0.0

srcMacAddress       = [0x00,0x00,0x00,0x01,0x00,0x01]
ControllerIpAddress = [192,168,10,2]
srcIpAddress        = [192,168,10,1]

PORT_SIM_TO_CTRL = 8001
PORT_SIM_TO_VIZ  = 8002
PORT_CTRL_TO_SIM = 8005

class Simulator:
    def __init__(self, args):
        self.componentId = 0
        self.localHost = args.server_url
        self.domain = args.domain
        self.portNum = 50101
        self.simulationStep = self.totalSimulationTime = 0
        self.stopRequested = False
        self.hCtrl = 0 
        self.hViz  = 0 
        self.hRecv = 0   
        self.mySignals = MySignals()
        self.path  = make_path()
        self.state = [self.path[0,0]+SPAWN_OFFSET[0],
                      self.path[0,1]+SPAWN_OFFSET[1],
                      SPAWN_OFFSET[2]]

    def mainThread(self):
        dSession = vsiCommonPythonApi.connectToServer(
            self.localHost, self.domain, self.portNum, self.componentId)
        vsiEthernetPythonGateway.initialize(
            dSession, self.componentId, bytes(srcMacAddress), bytes(srcIpAddress))
        try:
            vsiCommonPythonApi.waitForReset()
            self.updateInternalVariables()
            if vsiCommonPythonApi.isStopRequested(): raise Exception("stopRequested")

            self.hCtrl = vsiEthernetPythonGateway.tcpListen(PORT_SIM_TO_CTRL)
            self.hViz  = vsiEthernetPythonGateway.tcpListen(PORT_SIM_TO_VIZ)
            self.hRecv = vsiEthernetPythonGateway.tcpConnect(
                bytes(ControllerIpAddress), PORT_CTRL_TO_SIM)
            print(f"[Simulator] Connected  hCtrl={self.hCtrl} hViz={self.hViz} hRecv={self.hRecv}")

            nextExpectedTime = vsiCommonPythonApi.getSimulationTimeInNs()
            while vsiCommonPythonApi.getSimulationTimeInNs() < self.totalSimulationTime:
                self.updateInternalVariables()
                if vsiCommonPythonApi.isStopRequested(): raise Exception("stopRequested")
                if vsiEthernetPythonGateway.isTerminationOnGoing(): break
                if vsiEthernetPythonGateway.isTerminated(): break

                rd = vsiEthernetPythonGateway.recvEthernetPacket(PORT_CTRL_TO_SIM)
                if rd[3] != 0: self.decapsulate(rd)

                self.stepKinematics()

                data = (self.pack('d',self.mySignals.x)
                      + self.pack('d',self.mySignals.y)
                      + self.pack('d',self.mySignals.theta))
                vsiEthernetPythonGateway.sendEthernetPacket(self.hCtrl, bytes(data))
                vsiEthernetPythonGateway.sendEthernetPacket(self.hViz,  bytes(data))

                print(f"\n+=Simulator+=  t={vsiCommonPythonApi.getSimulationTimeInNs()} ns")
                print(f"  in : v={self.mySignals.v:.3f}  omega={self.mySignals.omega:.3f}")
                print(f"  out: x={self.mySignals.x:.3f}  y={self.mySignals.y:.3f}  theta={self.mySignals.theta:.3f}")

                self.updateInternalVariables()
                if vsiCommonPythonApi.isStopRequested(): raise Exception("stopRequested")
                nextExpectedTime += self.simulationStep
                if vsiCommonPythonApi.getSimulationTimeInNs() >= nextExpectedTime: continue
                if nextExpectedTime > self.totalSimulationTime:
                    vsiCommonPythonApi.advanceSimulation(self.totalSimulationTime - vsiCommonPythonApi.getSimulationTimeInNs()); break
                vsiCommonPythonApi.advanceSimulation(nextExpectedTime - vsiCommonPythonApi.getSimulationTimeInNs())

            if vsiCommonPythonApi.getSimulationTimeInNs() < self.totalSimulationTime:
                vsiEthernetPythonGateway.terminate()
        except Exception as e:
            if str(e) == "stopRequested": vsiCommonPythonApi.advanceSimulation(self.simulationStep+1)
            else: print(f"Simulator error: {e}")
        except: vsiCommonPythonApi.advanceSimulation(self.simulationStep+1)

    def stepKinematics(self):
        dt = max(self.simulationStep/1e9, 1e-6)
        x,y,th = self.state
        x  += self.mySignals.v * math.cos(th) * dt
        y  += self.mySignals.v * math.sin(th) * dt
        th += (self.mySignals.omega + DISTURBANCE) * dt
        if NOISE_STD > 0:
            x += np.random.normal(0,NOISE_STD); y += np.random.normal(0,NOISE_STD)
            th += np.random.normal(0,NOISE_STD*0.5)
        self.state=[x,y,th]
        self.mySignals.x=x; self.mySignals.y=y; self.mySignals.theta=th

    def decapsulate(self, rd):
        p = bytes(rd[2][:rd[3]])
        self.mySignals.v,     p = self.unpack('d',p)
        self.mySignals.omega, p = self.unpack('d',p)

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
    Simulator(p.parse_args()).mainThread()

if __name__=='__main__': main()
