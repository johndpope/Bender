//
//  ViewController.swift
//  Example
//
//  Copyright © 2017 Xmartlabs SRL. All rights reserved.
//

import Accelerate
import MetalPerformanceShaders
import MetalKit
import Bender
import UIKit

class MagentaViewController: UIViewController, ExampleViewController {

    var magentaNet: Network!
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    let inputSize = LayerSize(f: 3, w: 256)

    var pixelBufferPool: CVPixelBufferPool?
    @IBOutlet weak var imageView: UIImageView!

    override func viewDidLoad() {
        super.viewDidLoad()
        self.device = MTLCreateSystemDefaultDevice()!
        self.commandQueue = device.makeCommandQueue()
        setupMagenta()
        var me = self
        me.setPixelBufferPool()
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        runNetwork(self)
    }

    func setupMagenta() {
        measure("Set up takes:") {

            
            if let path = Bundle.main.path(forResource: "attention_rnn", ofType: "mag"){
                
                let model = try? Data(contentsOf:URL(fileURLWithPath: path))
                
                // pull apart magenta model.
                let myGraphProto = try? Tensorflow_Magenta_GeneratorBundle.init(serializedData:model!)
                
                let metaGraphData = myGraphProto!.metagraphFile
                
                if let metaGraphDef = try? Tensorflow_MetaGraphDef.init(serializedData:metaGraphData){
                    magentaNet = Network(device: device, inputSize: inputSize, parameterLoader: nil)

                    let converter = TFConverter.default()
                    converter.optimizers.append(TFInstanceNormOptimizer())
                    
                    
                    magentaNet.addPostProcessing(layers: [ImageLinearTransform()])
                    magentaNet.initialize()
                    
                    magentaNet.convertMetaGraph(converter: converter,metagraphFile:metaGraphDef)
                }
            }
            
          
        }
    }

    @IBAction func runNetwork(_ sender: Any) {
        let buffer = commandQueue.makeCommandBuffer()
        let image = loadTestImage(device: device, commandBuffer: buffer)
        buffer.commit()
        buffer.waitUntilCompleted()
        magentaNet.run(inputImage: image, queue: commandQueue) { [weak self] imageA in
            if let buffer = self?.getPixelBuffer(from: imageA.texture, bufferPool: self!.pixelBufferPool!) {
                let ciImage = CIImage(cvImageBuffer: buffer)
                let context = CIContext()
                let cgImage = context.createCGImage(ciImage, from: ciImage.extent)
                let uiImage = UIImage(cgImage: cgImage!)
                DispatchQueue.main.async {
                    self?.imageView.image = uiImage
                }
            }
        }
    }

    func loadTestImage(device: MTLDevice, commandBuffer: MTLCommandBuffer) -> MPSImage{
        // INPUT IMAGE
        let textureLoader = MTKTextureLoader(device: device)
        let inputTexture = try! textureLoader.newTexture(withContentsOf: Bundle.main.url(forResource: "wall-e", withExtension: "png")!, options: [MTKTextureLoaderOptionSRGB : NSNumber(value: false)])
        return MPSImage(texture: inputTexture, featureChannels: 3)
    }

    @IBAction func stopDebug(_ sender: Any) {
        commandQueue.insertDebugCaptureBoundary()
    }

}
