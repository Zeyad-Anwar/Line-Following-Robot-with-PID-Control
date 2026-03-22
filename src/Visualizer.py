from __future__ import print_function
import struct, sys, argparse, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

srcMacAddress       = [0x00,0x00,0x00,0x03,0x00,0x01]
SimulatorIpAddress  = [192,168,10,1]
ControllerIpAddress = [192,168,10,2]
srcIpAddress        = [192,168,10,3]

PORT_SIM_TO_VIZ  = 8002
PORT_CTRL_TO_VIZ = 8006

PLOT_OUTPUT = 'vsi_trajectory.png'

class Visualizer:
    def __init__(self, args):
        self.componentId=2
        self.localHost=args.server_url
        self.domain=args.domain
        self.portNum=50103
        self.simulationStep=self.totalSimulationTime=0
        self.stopRequested=False
        self.hSim =0
        self.hCtrl=0
        self.mySignals=MySignals()
        self.poses=[]; self.lat_errors=[]
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

            self.hSim  = vsiEthernetPythonGateway.tcpConnect(
                bytes(SimulatorIpAddress),  PORT_SIM_TO_VIZ)
            self.hCtrl = vsiEthernetPythonGateway.tcpConnect(
                bytes(ControllerIpAddress), PORT_CTRL_TO_VIZ)
            print(f"[Visualizer] Connected  hSim={self.hSim} hCtrl={self.hCtrl}")

            nextExpectedTime=vsiCommonPythonApi.getSimulationTimeInNs()
            while vsiCommonPythonApi.getSimulationTimeInNs()<self.totalSimulationTime:
                self.updateInternalVariables()
                if vsiCommonPythonApi.isStopRequested(): raise Exception("stopRequested")
                if vsiEthernetPythonGateway.isTerminationOnGoing(): break
                if vsiEthernetPythonGateway.isTerminated(): break

                rd=vsiEthernetPythonGateway.recvEthernetPacket(PORT_SIM_TO_VIZ)
                if rd[3]!=0: self.decapsulate(rd,'pose')
                rd=vsiEthernetPythonGateway.recvEthernetPacket(PORT_CTRL_TO_VIZ)
                if rd[3]!=0: self.decapsulate(rd,'cmdvel')

                self.poses.append((self.mySignals.x,self.mySignals.y,self.mySignals.theta))
                self.lat_errors.append(self.mySignals.lat_err)

                print(f"\n+=Visualizer+=  t={vsiCommonPythonApi.getSimulationTimeInNs()} ns")
                print(f"  pose: x={self.mySignals.x:.3f}  y={self.mySignals.y:.3f}  theta={self.mySignals.theta:.3f}")
                print(f"  err : lat={self.mySignals.lat_err:.3f}  head={self.mySignals.head_err:.3f}")

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
            else: print(f"Visualizer error: {e}")
        except: vsiCommonPythonApi.advanceSimulation(self.simulationStep+1)
        finally: self.savePlots()

    def decapsulate(self,rd,kind):
        p=bytes(rd[2][:rd[3]])
        if kind=='pose':
            self.mySignals.x,p=self.unpack('d',p)
            self.mySignals.y,p=self.unpack('d',p)
            self.mySignals.theta,p=self.unpack('d',p)
        else:
            self.mySignals.v,p=self.unpack('d',p)
            self.mySignals.omega,p=self.unpack('d',p)
            self.mySignals.lat_err,p=self.unpack('d',p)
            self.mySignals.head_err,p=self.unpack('d',p)

    def savePlots(self):
        if not self.poses: print("[Visualizer] No data to plot."); return
        arr=np.array(self.lat_errors)
        dt=max(self.simulationStep/1e9,0.05)
        overshoot=float(np.max(np.abs(arr)))
        settled=next((i+1 for i in range(len(arr)-1,-1,-1) if abs(arr[i])>0.05),0)
        ss=float(np.mean(np.abs(arr[int(0.8*len(arr)):])))
        print(f"\n[Visualizer] KPIs: OS={overshoot:.3f}m  Ts={settled*dt:.1f}s  SS={ss:.3f}m")

        fig,axes=plt.subplots(1,2,figsize=(13,4))
        fig.suptitle("VSI Line-Following Robot",fontweight='bold')
        ax=axes[0]
        ax.plot(self.path[:,0],self.path[:,1],'k--',lw=1.5,label='Reference')
        xs=[p[0] for p in self.poses]; ys=[p[1] for p in self.poses]
        ax.plot(xs,ys,'b-',lw=1.2,label='Robot')
        ax.plot(xs[0],ys[0],'go',ms=6); ax.plot(xs[-1],ys[-1],'rs',ms=6)
        ax.set_aspect('equal'); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.set_title('Trajectory'); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

        ax=axes[1]
        t=np.arange(len(arr))*dt
        ax.plot(t,arr,'b-',lw=0.9)
        ax.axhline(0,color='k',lw=0.8,ls='--')
        ax.axhline( 0.05,color='gray',lw=0.5,ls=':')
        ax.axhline(-0.05,color='gray',lw=0.5,ls=':',label='±5 cm')
        ax.text(0.98,0.95,f"OS={overshoot:.3f}m\nTs={settled*dt:.1f}s\nSS={ss:.3f}m",
                transform=ax.transAxes,fontsize=7,va='top',ha='right',
                bbox=dict(boxstyle='round,pad=0.3',fc='white',alpha=0.8))
        ax.set_xlabel('Time [s]'); ax.set_ylabel('Lateral error [m]')
        ax.set_title('Lateral Error'); ax.legend(fontsize=7); ax.grid(True,alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOT_OUTPUT,dpi=150,bbox_inches='tight')
        plt.close(fig)
        print(f"[Visualizer] Plot saved → {PLOT_OUTPUT}")

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
    Visualizer(p.parse_args()).mainThread()

if __name__=='__main__': main()
