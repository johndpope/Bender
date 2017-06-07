//
//  UnitTestViewController.swift
//  Example
//
//  Created by Mathias Claassen on 5/26/17.
//
//

import UIKit
import Bender

class TestViewController: UIViewController {

    let testRunner = BenderTestRunner()
    
    override func viewDidLoad() {
        super.viewDidLoad()
      //  testRunner.run()
        
        if let path = Bundle.main.path(forResource: "attention_rnn", ofType: "mag"){
            
            let model = try? Data(contentsOf:URL(fileURLWithPath: path))
            
            // pull apart inception model.
            let myGraphProto = try? Tensorflow_Magenta_GeneratorBundle.init(serializedData:model!)
            
            let metaGraphData = myGraphProto!.metagraphFile
            
            let graph1 = try? Tensorflow_MetaGraphDef.init(serializedData:metaGraphData)
            print("graph1:",graph1)
        }
       
        
    }

}
