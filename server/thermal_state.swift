import Foundation

switch ProcessInfo.processInfo.thermalState {
case .nominal: print(0)
case .fair: print(1)
case .serious: print(2)
case .critical: print(3)
@unknown default: print(-1)
}
