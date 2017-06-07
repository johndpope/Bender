// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorflow/core/framework/graph_transfer_info.proto
//
// For information on using the generated types, please see the documenation:
//   https://github.com/apple/swift-protobuf/

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

/// Protocol buffer representing a handle to a tensorflow resource. Handles are
/// not valid across executions, but can be serialized back and forth from within
/// a single run.
public struct Tensorflow_GraphTransferInfo: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".GraphTransferInfo"

  public var nodeInfo: [Tensorflow_GraphTransferInfo.NodeInfo] = []

  public var constNodeInfo: [Tensorflow_GraphTransferInfo.ConstNodeInfo] = []

  public var nodeInputInfo: [Tensorflow_GraphTransferInfo.NodeInputInfo] = []

  public var nodeOutputInfo: [Tensorflow_GraphTransferInfo.NodeOutputInfo] = []

  /// Input Node parameters of transferred graph
  public var graphInputNodeInfo: [Tensorflow_GraphTransferInfo.GraphInputNodeInfo] = []

  public var graphOutputNodeInfo: [Tensorflow_GraphTransferInfo.GraphOutputNodeInfo] = []

  /// Destination of graph transfer
  public var destination: Tensorflow_GraphTransferInfo.Destination = .nop

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public enum Destination: SwiftProtobuf.Enum {
    public typealias RawValue = Int
    case nop // = 0
    case hexagon // = 1
    case UNRECOGNIZED(Int)

    public init() {
      self = .nop
    }

    public init?(rawValue: Int) {
      switch rawValue {
      case 0: self = .nop
      case 1: self = .hexagon
      default: self = .UNRECOGNIZED(rawValue)
      }
    }

    public var rawValue: Int {
      switch self {
      case .nop: return 0
      case .hexagon: return 1
      case .UNRECOGNIZED(let i): return i
      }
    }

  }

  public struct NodeInput: SwiftProtobuf.Message {
    public static let protoMessageName: String = Tensorflow_GraphTransferInfo.protoMessageName + ".NodeInput"

    public var nodeID: Int32 = 0

    public var outputPort: Int32 = 0

    public var unknownFields = SwiftProtobuf.UnknownStorage()

    public init() {}

    /// Used by the decoding initializers in the SwiftProtobuf library, not generally
    /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
    /// initializers are defined in the SwiftProtobuf library. See the Message and
    /// Message+*Additions` files.
    public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
      while let fieldNumber = try decoder.nextFieldNumber() {
        switch fieldNumber {
        case 1: try decoder.decodeSingularInt32Field(value: &self.nodeID)
        case 2: try decoder.decodeSingularInt32Field(value: &self.outputPort)
        default: break
        }
      }
    }

    /// Used by the encoding methods of the SwiftProtobuf library, not generally
    /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
    /// other serializer methods are defined in the SwiftProtobuf library. See the
    /// `Message` and `Message+*Additions` files.
    public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
      if self.nodeID != 0 {
        try visitor.visitSingularInt32Field(value: self.nodeID, fieldNumber: 1)
      }
      if self.outputPort != 0 {
        try visitor.visitSingularInt32Field(value: self.outputPort, fieldNumber: 2)
      }
      try unknownFields.traverse(visitor: &visitor)
    }
  }

  public struct NodeInfo: SwiftProtobuf.Message {
    public static let protoMessageName: String = Tensorflow_GraphTransferInfo.protoMessageName + ".NodeInfo"

    public var name: String = String()

    public var nodeID: Int32 = 0

    public var typeName: String = String()

    public var socOpID: Int32 = 0

    public var paddingID: Int32 = 0

    public var inputCount: Int32 = 0

    public var outputCount: Int32 = 0

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
        case 2: try decoder.decodeSingularInt32Field(value: &self.nodeID)
        case 3: try decoder.decodeSingularStringField(value: &self.typeName)
        case 4: try decoder.decodeSingularInt32Field(value: &self.socOpID)
        case 5: try decoder.decodeSingularInt32Field(value: &self.paddingID)
        case 6: try decoder.decodeSingularInt32Field(value: &self.inputCount)
        case 7: try decoder.decodeSingularInt32Field(value: &self.outputCount)
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
      if self.nodeID != 0 {
        try visitor.visitSingularInt32Field(value: self.nodeID, fieldNumber: 2)
      }
      if !self.typeName.isEmpty {
        try visitor.visitSingularStringField(value: self.typeName, fieldNumber: 3)
      }
      if self.socOpID != 0 {
        try visitor.visitSingularInt32Field(value: self.socOpID, fieldNumber: 4)
      }
      if self.paddingID != 0 {
        try visitor.visitSingularInt32Field(value: self.paddingID, fieldNumber: 5)
      }
      if self.inputCount != 0 {
        try visitor.visitSingularInt32Field(value: self.inputCount, fieldNumber: 6)
      }
      if self.outputCount != 0 {
        try visitor.visitSingularInt32Field(value: self.outputCount, fieldNumber: 7)
      }
      try unknownFields.traverse(visitor: &visitor)
    }
  }

  public struct ConstNodeInfo: SwiftProtobuf.Message {
    public static let protoMessageName: String = Tensorflow_GraphTransferInfo.protoMessageName + ".ConstNodeInfo"

    public var name: String = String()

    public var nodeID: Int32 = 0

    public var shape: [Int64] = []

    public var data: Data = SwiftProtobuf.Internal.emptyData

    public var dtype: Tensorflow_DataType = .dtInvalid

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
        case 2: try decoder.decodeSingularInt32Field(value: &self.nodeID)
        case 3: try decoder.decodeRepeatedInt64Field(value: &self.shape)
        case 4: try decoder.decodeSingularBytesField(value: &self.data)
        case 5: try decoder.decodeSingularEnumField(value: &self.dtype)
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
      if self.nodeID != 0 {
        try visitor.visitSingularInt32Field(value: self.nodeID, fieldNumber: 2)
      }
      if !self.shape.isEmpty {
        try visitor.visitPackedInt64Field(value: self.shape, fieldNumber: 3)
      }
      if !self.data.isEmpty {
        try visitor.visitSingularBytesField(value: self.data, fieldNumber: 4)
      }
      if self.dtype != .dtInvalid {
        try visitor.visitSingularEnumField(value: self.dtype, fieldNumber: 5)
      }
      try unknownFields.traverse(visitor: &visitor)
    }
  }

  public struct NodeInputInfo: SwiftProtobuf.Message {
    public static let protoMessageName: String = Tensorflow_GraphTransferInfo.protoMessageName + ".NodeInputInfo"

    public var nodeID: Int32 = 0

    public var nodeInput: [Tensorflow_GraphTransferInfo.NodeInput] = []

    public var unknownFields = SwiftProtobuf.UnknownStorage()

    public init() {}

    /// Used by the decoding initializers in the SwiftProtobuf library, not generally
    /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
    /// initializers are defined in the SwiftProtobuf library. See the Message and
    /// Message+*Additions` files.
    public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
      while let fieldNumber = try decoder.nextFieldNumber() {
        switch fieldNumber {
        case 1: try decoder.decodeSingularInt32Field(value: &self.nodeID)
        case 2: try decoder.decodeRepeatedMessageField(value: &self.nodeInput)
        default: break
        }
      }
    }

    /// Used by the encoding methods of the SwiftProtobuf library, not generally
    /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
    /// other serializer methods are defined in the SwiftProtobuf library. See the
    /// `Message` and `Message+*Additions` files.
    public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
      if self.nodeID != 0 {
        try visitor.visitSingularInt32Field(value: self.nodeID, fieldNumber: 1)
      }
      if !self.nodeInput.isEmpty {
        try visitor.visitRepeatedMessageField(value: self.nodeInput, fieldNumber: 2)
      }
      try unknownFields.traverse(visitor: &visitor)
    }
  }

  public struct NodeOutputInfo: SwiftProtobuf.Message {
    public static let protoMessageName: String = Tensorflow_GraphTransferInfo.protoMessageName + ".NodeOutputInfo"

    public var nodeID: Int32 = 0

    public var maxByteSize: [Int32] = []

    public var unknownFields = SwiftProtobuf.UnknownStorage()

    public init() {}

    /// Used by the decoding initializers in the SwiftProtobuf library, not generally
    /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
    /// initializers are defined in the SwiftProtobuf library. See the Message and
    /// Message+*Additions` files.
    public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
      while let fieldNumber = try decoder.nextFieldNumber() {
        switch fieldNumber {
        case 1: try decoder.decodeSingularInt32Field(value: &self.nodeID)
        case 2: try decoder.decodeRepeatedInt32Field(value: &self.maxByteSize)
        default: break
        }
      }
    }

    /// Used by the encoding methods of the SwiftProtobuf library, not generally
    /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
    /// other serializer methods are defined in the SwiftProtobuf library. See the
    /// `Message` and `Message+*Additions` files.
    public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
      if self.nodeID != 0 {
        try visitor.visitSingularInt32Field(value: self.nodeID, fieldNumber: 1)
      }
      if !self.maxByteSize.isEmpty {
        try visitor.visitPackedInt32Field(value: self.maxByteSize, fieldNumber: 2)
      }
      try unknownFields.traverse(visitor: &visitor)
    }
  }

  public struct GraphInputNodeInfo: SwiftProtobuf.Message {
    public static let protoMessageName: String = Tensorflow_GraphTransferInfo.protoMessageName + ".GraphInputNodeInfo"

    public var name: String = String()

    public var shape: [Int64] = []

    public var dtype: Tensorflow_DataType = .dtInvalid

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
        case 2: try decoder.decodeRepeatedInt64Field(value: &self.shape)
        case 3: try decoder.decodeSingularEnumField(value: &self.dtype)
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
      if !self.shape.isEmpty {
        try visitor.visitPackedInt64Field(value: self.shape, fieldNumber: 2)
      }
      if self.dtype != .dtInvalid {
        try visitor.visitSingularEnumField(value: self.dtype, fieldNumber: 3)
      }
      try unknownFields.traverse(visitor: &visitor)
    }
  }

  public struct GraphOutputNodeInfo: SwiftProtobuf.Message {
    public static let protoMessageName: String = Tensorflow_GraphTransferInfo.protoMessageName + ".GraphOutputNodeInfo"

    public var name: String = String()

    public var shape: [Int64] = []

    public var dtype: Tensorflow_DataType = .dtInvalid

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
        case 2: try decoder.decodeRepeatedInt64Field(value: &self.shape)
        case 3: try decoder.decodeSingularEnumField(value: &self.dtype)
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
      if !self.shape.isEmpty {
        try visitor.visitPackedInt64Field(value: self.shape, fieldNumber: 2)
      }
      if self.dtype != .dtInvalid {
        try visitor.visitSingularEnumField(value: self.dtype, fieldNumber: 3)
      }
      try unknownFields.traverse(visitor: &visitor)
    }
  }

  public init() {}

  /// Used by the decoding initializers in the SwiftProtobuf library, not generally
  /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
  /// initializers are defined in the SwiftProtobuf library. See the Message and
  /// Message+*Additions` files.
  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeRepeatedMessageField(value: &self.nodeInfo)
      case 2: try decoder.decodeRepeatedMessageField(value: &self.constNodeInfo)
      case 3: try decoder.decodeRepeatedMessageField(value: &self.nodeInputInfo)
      case 4: try decoder.decodeRepeatedMessageField(value: &self.nodeOutputInfo)
      case 5: try decoder.decodeRepeatedMessageField(value: &self.graphInputNodeInfo)
      case 6: try decoder.decodeRepeatedMessageField(value: &self.graphOutputNodeInfo)
      case 7: try decoder.decodeSingularEnumField(value: &self.destination)
      default: break
      }
    }
  }

  /// Used by the encoding methods of the SwiftProtobuf library, not generally
  /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
  /// other serializer methods are defined in the SwiftProtobuf library. See the
  /// `Message` and `Message+*Additions` files.
  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if !self.nodeInfo.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.nodeInfo, fieldNumber: 1)
    }
    if !self.constNodeInfo.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.constNodeInfo, fieldNumber: 2)
    }
    if !self.nodeInputInfo.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.nodeInputInfo, fieldNumber: 3)
    }
    if !self.nodeOutputInfo.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.nodeOutputInfo, fieldNumber: 4)
    }
    if !self.graphInputNodeInfo.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.graphInputNodeInfo, fieldNumber: 5)
    }
    if !self.graphOutputNodeInfo.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.graphOutputNodeInfo, fieldNumber: 6)
    }
    if self.destination != .nop {
      try visitor.visitSingularEnumField(value: self.destination, fieldNumber: 7)
    }
    try unknownFields.traverse(visitor: &visitor)
  }
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow"

extension Tensorflow_GraphTransferInfo: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "node_info"),
    2: .standard(proto: "const_node_info"),
    3: .standard(proto: "node_input_info"),
    4: .standard(proto: "node_output_info"),
    5: .standard(proto: "graph_input_node_info"),
    6: .standard(proto: "graph_output_node_info"),
    7: .same(proto: "destination"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_GraphTransferInfo) -> Bool {
    if self.nodeInfo != other.nodeInfo {return false}
    if self.constNodeInfo != other.constNodeInfo {return false}
    if self.nodeInputInfo != other.nodeInputInfo {return false}
    if self.nodeOutputInfo != other.nodeOutputInfo {return false}
    if self.graphInputNodeInfo != other.graphInputNodeInfo {return false}
    if self.graphOutputNodeInfo != other.graphOutputNodeInfo {return false}
    if self.destination != other.destination {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_GraphTransferInfo.Destination: SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    0: .same(proto: "NOP"),
    1: .same(proto: "HEXAGON"),
  ]
}

extension Tensorflow_GraphTransferInfo.NodeInput: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "node_id"),
    2: .standard(proto: "output_port"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_GraphTransferInfo.NodeInput) -> Bool {
    if self.nodeID != other.nodeID {return false}
    if self.outputPort != other.outputPort {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_GraphTransferInfo.NodeInfo: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "name"),
    2: .standard(proto: "node_id"),
    3: .standard(proto: "type_name"),
    4: .standard(proto: "soc_op_id"),
    5: .standard(proto: "padding_id"),
    6: .standard(proto: "input_count"),
    7: .standard(proto: "output_count"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_GraphTransferInfo.NodeInfo) -> Bool {
    if self.name != other.name {return false}
    if self.nodeID != other.nodeID {return false}
    if self.typeName != other.typeName {return false}
    if self.socOpID != other.socOpID {return false}
    if self.paddingID != other.paddingID {return false}
    if self.inputCount != other.inputCount {return false}
    if self.outputCount != other.outputCount {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_GraphTransferInfo.ConstNodeInfo: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "name"),
    2: .standard(proto: "node_id"),
    3: .same(proto: "shape"),
    4: .same(proto: "data"),
    5: .same(proto: "dtype"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_GraphTransferInfo.ConstNodeInfo) -> Bool {
    if self.name != other.name {return false}
    if self.nodeID != other.nodeID {return false}
    if self.shape != other.shape {return false}
    if self.data != other.data {return false}
    if self.dtype != other.dtype {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_GraphTransferInfo.NodeInputInfo: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "node_id"),
    2: .standard(proto: "node_input"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_GraphTransferInfo.NodeInputInfo) -> Bool {
    if self.nodeID != other.nodeID {return false}
    if self.nodeInput != other.nodeInput {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_GraphTransferInfo.NodeOutputInfo: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "node_id"),
    2: .standard(proto: "max_byte_size"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_GraphTransferInfo.NodeOutputInfo) -> Bool {
    if self.nodeID != other.nodeID {return false}
    if self.maxByteSize != other.maxByteSize {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_GraphTransferInfo.GraphInputNodeInfo: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "name"),
    2: .same(proto: "shape"),
    3: .same(proto: "dtype"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_GraphTransferInfo.GraphInputNodeInfo) -> Bool {
    if self.name != other.name {return false}
    if self.shape != other.shape {return false}
    if self.dtype != other.dtype {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_GraphTransferInfo.GraphOutputNodeInfo: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "name"),
    2: .same(proto: "shape"),
    3: .same(proto: "dtype"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_GraphTransferInfo.GraphOutputNodeInfo) -> Bool {
    if self.name != other.name {return false}
    if self.shape != other.shape {return false}
    if self.dtype != other.dtype {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}
