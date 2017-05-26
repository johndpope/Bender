//
//  Network.swift
//  Palladium
//
//  Created by Mathias Claassen on 5/4/17.
//
//

import MetalPerformanceShaders


/// Represents a neural network
open class Network {

    public var start: Start
    var nodes = [NetworkLayer]()
    fileprivate var device: MTLDevice
    public var parameterLoader: ParameterLoader

    ///
    /// - Parameters:
    ///   - device: the MTLDevice.
    ///   - inputSize: The image size for the first layer. Input images will be resized if they do not have this size.
    ///   - parameterLoader: The parameter loader responsible for loading the weights and biases for this network.
    public init(device: MTLDevice, inputSize: LayerSize, parameterLoader: ParameterLoader) {
        start = Start(size: inputSize)
        self.device = device
        self.parameterLoader = parameterLoader
    }

    open func initialize() {
        buildExecutionList(node: start)
        for layer in nodes {
            layer.initialize(network: self, device: device)
        }
        nodes = nodes.filter { !($0 is Dummy) }
        _ = nodes.map {
            print($0.id ?? "nil")
        }
    }

    public func run(inputImage: MPSImage, queue: MTLCommandQueue, result: @escaping (MPSImage) -> Void) {

        queue.insertDebugCaptureBoundary() // DEBUG
        let commandBuffer = queue.makeCommandBuffer()
        commandBuffer.label = "Network run buffer"
        start.inputImage = inputImage
        autoreleasepool {
            for layer in nodes {
                layer.execute(commandBuffer: commandBuffer)
            }
            commandBuffer.commit()
            //TODO: We should execute this on another dispatch queue
            commandBuffer.waitUntilCompleted()
            result(nodes.last!.outputImage)
        }
    }


    /// Update weights of the network.
    ///
    public func change(to checkpoint: String) {
        if checkpoint == parameterLoader.checkpoint {
            return
        }

        parameterLoader.checkpoint = checkpoint
        for layer in nodes {
            layer.updatedCheckpoint(device: device)
        }
    }

    func buildExecutionList(node: NetworkLayer) {
        guard !node.getIncoming().contains (where: { incoming in
            return !nodes.contains(incoming)
        }) else { return }
        nodes.append(node)
        for node in node.getOutgoing() {
            buildExecutionList(node: node)
        }
    }

}