// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorflow/contrib/tensorboard/plugins/trace/trace_info.proto
//
// For information on using the generated types, please see the documenation:
//   https://github.com/apple/swift-protobuf/

// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
//
//Licensed under the Apache License, Version 2.0 (the "License");
//you may not use this file except in compliance with the License.
//You may obtain a copy of the License at
//
//http://www.apache.org/licenses/LICENSE-2.0
//
//Unless required by applicable law or agreed to in writing, software
//distributed under the License is distributed on an "AS IS" BASIS,
//WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//See the License for the specific language governing permissions and
//limitations under the License.
//==============================================================================

import Foundation
import SwiftProtobuf

// If the compiler emits an error on this type, it is because this file
// was generated by a version of the `protoc` Swift plug-in that is
// incompatible with the version of SwiftProtobuf to which you are linking.
// Please ensure that your are building against the same version of the API
// that was used to generate this file.
fileprivate struct _GeneratedWithProtocGenSwiftVersion: SwiftProtobuf.ProtobufAPIVersionCheck {
  struct _2: SwiftProtobuf.ProtobufAPIVersion_2 {}
  typealias Version = _2
}

public struct Tensorflow_Contrib_Tensorboard_TraceInfo: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".TraceInfo"

  public var ops: [Tensorflow_Contrib_Tensorboard_OpInfo] = []

  public var files: [Tensorflow_Contrib_Tensorboard_FileInfo] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  /// Used by the decoding initializers in the SwiftProtobuf library, not generally
  /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
  /// initializers are defined in the SwiftProtobuf library. See the Message and
  /// Message+*Additions` files.
  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedMessageField(value: &self.ops)
      case 2: try decoder.decodeRepeatedMessageField(value: &self.files)
      default: break
      }
    }
  }

  /// Used by the encoding methods of the SwiftProtobuf library, not generally
  /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
  /// other serializer methods are defined in the SwiftProtobuf library. See the
  /// `Message` and `Message+*Additions` files.
  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.ops.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.ops, fieldNumber: 1)
    }
    if !self.files.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.files, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }
}

public struct Tensorflow_Contrib_Tensorboard_OpInfo: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".OpInfo"

  public var name: String = String()

  public var opType: String = String()

  public var device: String = String()

  public var traceback: [Tensorflow_Contrib_Tensorboard_LineTrace] = []

  public var inputs: [Tensorflow_Contrib_Tensorboard_TensorInfo] = []

  public var outputs: [Tensorflow_Contrib_Tensorboard_TensorInfo] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  /// Used by the decoding initializers in the SwiftProtobuf library, not generally
  /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
  /// initializers are defined in the SwiftProtobuf library. See the Message and
  /// Message+*Additions` files.
  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self.name)
      case 2: try decoder.decodeSingularStringField(value: &self.opType)
      case 3: try decoder.decodeSingularStringField(value: &self.device)
      case 4: try decoder.decodeRepeatedMessageField(value: &self.traceback)
      case 5: try decoder.decodeRepeatedMessageField(value: &self.inputs)
      case 6: try decoder.decodeRepeatedMessageField(value: &self.outputs)
      default: break
      }
    }
  }

  /// Used by the encoding methods of the SwiftProtobuf library, not generally
  /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
  /// other serializer methods are defined in the SwiftProtobuf library. See the
  /// `Message` and `Message+*Additions` files.
  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.name.isEmpty {
      try visitor.visitSingularStringField(value: self.name, fieldNumber: 1)
    }
    if !self.opType.isEmpty {
      try visitor.visitSingularStringField(value: self.opType, fieldNumber: 2)
    }
    if !self.device.isEmpty {
      try visitor.visitSingularStringField(value: self.device, fieldNumber: 3)
    }
    if !self.traceback.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.traceback, fieldNumber: 4)
    }
    if !self.inputs.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.inputs, fieldNumber: 5)
    }
    if !self.outputs.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.outputs, fieldNumber: 6)
    }
    try unknownFields.traverse(visitor: &visitor)
  }
}

public struct Tensorflow_Contrib_Tensorboard_LineTrace: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".LineTrace"

  /// Absolute file path.
  public var filePath: String = String()

  /// 1-based line number.
  public var lineNumber: UInt32 = 0

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  /// Used by the decoding initializers in the SwiftProtobuf library, not generally
  /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
  /// initializers are defined in the SwiftProtobuf library. See the Message and
  /// Message+*Additions` files.
  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self.filePath)
      case 2: try decoder.decodeSingularUInt32Field(value: &self.lineNumber)
      default: break
      }
    }
  }

  /// Used by the encoding methods of the SwiftProtobuf library, not generally
  /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
  /// other serializer methods are defined in the SwiftProtobuf library. See the
  /// `Message` and `Message+*Additions` files.
  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.filePath.isEmpty {
      try visitor.visitSingularStringField(value: self.filePath, fieldNumber: 1)
    }
    if self.lineNumber != 0 {
      try visitor.visitSingularUInt32Field(value: self.lineNumber, fieldNumber: 2)
    }
    try unknownFields.traverse(visitor: &visitor)
  }
}

public struct Tensorflow_Contrib_Tensorboard_TensorInfo: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".TensorInfo"

  /// Size of the tensor for each dimension. Value of -1 denotes "unknown"
  /// size for that dimension.
  public var shape: [Int32] = []

  /// The data type of the tensor.
  public var dtype: String = String()

  /// Number of bytes per element in the tensor.
  public var numBytesPerElem: UInt32 = 0

  /// List of operation names that consume this tensor.
  public var consumers: [String] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  /// Used by the decoding initializers in the SwiftProtobuf library, not generally
  /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
  /// initializers are defined in the SwiftProtobuf library. See the Message and
  /// Message+*Additions` files.
  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedInt32Field(value: &self.shape)
      case 2: try decoder.decodeSingularStringField(value: &self.dtype)
      case 3: try decoder.decodeSingularUInt32Field(value: &self.numBytesPerElem)
      case 4: try decoder.decodeRepeatedStringField(value: &self.consumers)
      default: break
      }
    }
  }

  /// Used by the encoding methods of the SwiftProtobuf library, not generally
  /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
  /// other serializer methods are defined in the SwiftProtobuf library. See the
  /// `Message` and `Message+*Additions` files.
  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.shape.isEmpty {
      try visitor.visitPackedInt32Field(value: self.shape, fieldNumber: 1)
    }
    if !self.dtype.isEmpty {
      try visitor.visitSingularStringField(value: self.dtype, fieldNumber: 2)
    }
    if self.numBytesPerElem != 0 {
      try visitor.visitSingularUInt32Field(value: self.numBytesPerElem, fieldNumber: 3)
    }
    if !self.consumers.isEmpty {
      try visitor.visitRepeatedStringField(value: self.consumers, fieldNumber: 4)
    }
    try unknownFields.traverse(visitor: &visitor)
  }
}

public struct Tensorflow_Contrib_Tensorboard_FileInfo: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".FileInfo"

  /// Absolute file path to the source code.
  public var filePath: String = String()

  public var sourceCode: String = String()

  /// Map from end of statement to start of statement. End and start are 0-based
  /// line indexes.
  public var multilineStatements: Dictionary<UInt32,UInt32> = [:]

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  /// Used by the decoding initializers in the SwiftProtobuf library, not generally
  /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
  /// initializers are defined in the SwiftProtobuf library. See the Message and
  /// Message+*Additions` files.
  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self.filePath)
      case 2: try decoder.decodeSingularStringField(value: &self.sourceCode)
      case 3: try decoder.decodeMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufUInt32,SwiftProtobuf.ProtobufUInt32>.self, value: &self.multilineStatements)
      default: break
      }
    }
  }

  /// Used by the encoding methods of the SwiftProtobuf library, not generally
  /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
  /// other serializer methods are defined in the SwiftProtobuf library. See the
  /// `Message` and `Message+*Additions` files.
  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.filePath.isEmpty {
      try visitor.visitSingularStringField(value: self.filePath, fieldNumber: 1)
    }
    if !self.sourceCode.isEmpty {
      try visitor.visitSingularStringField(value: self.sourceCode, fieldNumber: 2)
    }
    if !self.multilineStatements.isEmpty {
      try visitor.visitMapField(fieldType: SwiftProtobuf._ProtobufMap<SwiftProtobuf.ProtobufUInt32,SwiftProtobuf.ProtobufUInt32>.self, value: self.multilineStatements, fieldNumber: 3)
    }
    try unknownFields.traverse(visitor: &visitor)
  }
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow.contrib.tensorboard"

extension Tensorflow_Contrib_Tensorboard_TraceInfo: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "ops"),
    2: .same(proto: "files"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Contrib_Tensorboard_TraceInfo) -> Bool {
    if self.ops != other.ops {return false}
    if self.files != other.files {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_Contrib_Tensorboard_OpInfo: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "name"),
    2: .standard(proto: "op_type"),
    3: .same(proto: "device"),
    4: .same(proto: "traceback"),
    5: .same(proto: "inputs"),
    6: .same(proto: "outputs"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Contrib_Tensorboard_OpInfo) -> Bool {
    if self.name != other.name {return false}
    if self.opType != other.opType {return false}
    if self.device != other.device {return false}
    if self.traceback != other.traceback {return false}
    if self.inputs != other.inputs {return false}
    if self.outputs != other.outputs {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_Contrib_Tensorboard_LineTrace: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "file_path"),
    2: .standard(proto: "line_number"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Contrib_Tensorboard_LineTrace) -> Bool {
    if self.filePath != other.filePath {return false}
    if self.lineNumber != other.lineNumber {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_Contrib_Tensorboard_TensorInfo: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "shape"),
    2: .same(proto: "dtype"),
    3: .standard(proto: "num_bytes_per_elem"),
    4: .same(proto: "consumers"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Contrib_Tensorboard_TensorInfo) -> Bool {
    if self.shape != other.shape {return false}
    if self.dtype != other.dtype {return false}
    if self.numBytesPerElem != other.numBytesPerElem {return false}
    if self.consumers != other.consumers {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_Contrib_Tensorboard_FileInfo: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "file_path"),
    2: .standard(proto: "source_code"),
    3: .standard(proto: "multiline_statements"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Contrib_Tensorboard_FileInfo) -> Bool {
    if self.filePath != other.filePath {return false}
    if self.sourceCode != other.sourceCode {return false}
    if self.multilineStatements != other.multilineStatements {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}
