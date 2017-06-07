// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorflow/tools/tfprof/tfprof_output.proto
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

public struct Tensorflow_Tfprof_TFProfTensorProto: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".TFProfTensorProto"

  public var dtype: Tensorflow_DataType {
    get {return _dtype ?? .dtInvalid}
    set {_dtype = newValue}
  }
  /// Returns true if `dtype` has been explicitly set.
  public var hasDtype: Bool {return self._dtype != nil}
  /// Clears the value of `dtype`. Subsequent reads from it will return its default value.
  public mutating func clearDtype() {self._dtype = nil}

  /// Flatten tensor in row-major.
  /// Only one of the following array is set.
  public var valueDouble: [Double] = []

  public var valueInt64: [Int64] = []

  public var valueStr: [String] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  /// Used by the decoding initializers in the SwiftProtobuf library, not generally
  /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
  /// initializers are defined in the SwiftProtobuf library. See the Message and
  /// Message+*Additions` files.
  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularEnumField(value: &self._dtype)
      case 2: try decoder.decodeRepeatedDoubleField(value: &self.valueDouble)
      case 3: try decoder.decodeRepeatedInt64Field(value: &self.valueInt64)
      case 4: try decoder.decodeRepeatedStringField(value: &self.valueStr)
      default: break
      }
    }
  }

  /// Used by the encoding methods of the SwiftProtobuf library, not generally
  /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
  /// other serializer methods are defined in the SwiftProtobuf library. See the
  /// `Message` and `Message+*Additions` files.
  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if let v = self._dtype {
      try visitor.visitSingularEnumField(value: v, fieldNumber: 1)
    }
    if !self.valueDouble.isEmpty {
      try visitor.visitRepeatedDoubleField(value: self.valueDouble, fieldNumber: 2)
    }
    if !self.valueInt64.isEmpty {
      try visitor.visitRepeatedInt64Field(value: self.valueInt64, fieldNumber: 3)
    }
    if !self.valueStr.isEmpty {
      try visitor.visitRepeatedStringField(value: self.valueStr, fieldNumber: 4)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  fileprivate var _dtype: Tensorflow_DataType? = nil
}

/// A node in TensorFlow graph. Used by scope/graph view.
public struct Tensorflow_Tfprof_TFGraphNodeProto: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".TFGraphNodeProto"

  /// op name.
  public var name: String {
    get {return _storage._name ?? String()}
    set {_uniqueStorage()._name = newValue}
  }
  /// Returns true if `name` has been explicitly set.
  public var hasName: Bool {return _storage._name != nil}
  /// Clears the value of `name`. Subsequent reads from it will return its default value.
  public mutating func clearName() {_storage._name = nil}

  /// tensor value restored from checkpoint.
  public var tensorValue: Tensorflow_Tfprof_TFProfTensorProto {
    get {return _storage._tensorValue ?? Tensorflow_Tfprof_TFProfTensorProto()}
    set {_uniqueStorage()._tensorValue = newValue}
  }
  /// Returns true if `tensorValue` has been explicitly set.
  public var hasTensorValue: Bool {return _storage._tensorValue != nil}
  /// Clears the value of `tensorValue`. Subsequent reads from it will return its default value.
  public mutating func clearTensorValue() {_storage._tensorValue = nil}

  /// op execution time.
  public var execMicros: Int64 {
    get {return _storage._execMicros ?? 0}
    set {_uniqueStorage()._execMicros = newValue}
  }
  /// Returns true if `execMicros` has been explicitly set.
  public var hasExecMicros: Bool {return _storage._execMicros != nil}
  /// Clears the value of `execMicros`. Subsequent reads from it will return its default value.
  public mutating func clearExecMicros() {_storage._execMicros = nil}

  /// Total requested bytes by the op.
  public var requestedBytes: Int64 {
    get {return _storage._requestedBytes ?? 0}
    set {_uniqueStorage()._requestedBytes = newValue}
  }
  /// Returns true if `requestedBytes` has been explicitly set.
  public var hasRequestedBytes: Bool {return _storage._requestedBytes != nil}
  /// Clears the value of `requestedBytes`. Subsequent reads from it will return its default value.
  public mutating func clearRequestedBytes() {_storage._requestedBytes = nil}

  /// Number of parameters if available.
  public var parameters: Int64 {
    get {return _storage._parameters ?? 0}
    set {_uniqueStorage()._parameters = newValue}
  }
  /// Returns true if `parameters` has been explicitly set.
  public var hasParameters: Bool {return _storage._parameters != nil}
  /// Clears the value of `parameters`. Subsequent reads from it will return its default value.
  public mutating func clearParameters() {_storage._parameters = nil}

  /// Number of float operations.
  public var floatOps: Int64 {
    get {return _storage._floatOps ?? 0}
    set {_uniqueStorage()._floatOps = newValue}
  }
  /// Returns true if `floatOps` has been explicitly set.
  public var hasFloatOps: Bool {return _storage._floatOps != nil}
  /// Clears the value of `floatOps`. Subsequent reads from it will return its default value.
  public mutating func clearFloatOps() {_storage._floatOps = nil}

  /// Device the op is assigned to.
  /// Since an op can fire multiple kernel calls, there can be multiple devices.
  public var devices: [String] {
    get {return _storage._devices}
    set {_uniqueStorage()._devices = newValue}
  }

  /// The following are the aggregated stats from all accounted children and
  /// the node itself. The actual children depend on the data structure used
  /// (scope, graph).
  public var totalExecMicros: Int64 {
    get {return _storage._totalExecMicros ?? 0}
    set {_uniqueStorage()._totalExecMicros = newValue}
  }
  /// Returns true if `totalExecMicros` has been explicitly set.
  public var hasTotalExecMicros: Bool {return _storage._totalExecMicros != nil}
  /// Clears the value of `totalExecMicros`. Subsequent reads from it will return its default value.
  public mutating func clearTotalExecMicros() {_storage._totalExecMicros = nil}

  public var totalRequestedBytes: Int64 {
    get {return _storage._totalRequestedBytes ?? 0}
    set {_uniqueStorage()._totalRequestedBytes = newValue}
  }
  /// Returns true if `totalRequestedBytes` has been explicitly set.
  public var hasTotalRequestedBytes: Bool {return _storage._totalRequestedBytes != nil}
  /// Clears the value of `totalRequestedBytes`. Subsequent reads from it will return its default value.
  public mutating func clearTotalRequestedBytes() {_storage._totalRequestedBytes = nil}

  public var totalParameters: Int64 {
    get {return _storage._totalParameters ?? 0}
    set {_uniqueStorage()._totalParameters = newValue}
  }
  /// Returns true if `totalParameters` has been explicitly set.
  public var hasTotalParameters: Bool {return _storage._totalParameters != nil}
  /// Clears the value of `totalParameters`. Subsequent reads from it will return its default value.
  public mutating func clearTotalParameters() {_storage._totalParameters = nil}

  public var totalFloatOps: Int64 {
    get {return _storage._totalFloatOps ?? 0}
    set {_uniqueStorage()._totalFloatOps = newValue}
  }
  /// Returns true if `totalFloatOps` has been explicitly set.
  public var hasTotalFloatOps: Bool {return _storage._totalFloatOps != nil}
  /// Clears the value of `totalFloatOps`. Subsequent reads from it will return its default value.
  public mutating func clearTotalFloatOps() {_storage._totalFloatOps = nil}

  /// shape information, if available.
  public var shapes: [Tensorflow_TensorShapeProto] {
    get {return _storage._shapes}
    set {_uniqueStorage()._shapes = newValue}
  }

  /// Descendants of the graph. The actual descendants depend on the data
  /// structure used (scope, graph).
  public var children: [Tensorflow_Tfprof_TFGraphNodeProto] {
    get {return _storage._children}
    set {_uniqueStorage()._children = newValue}
  }

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  /// Used by the decoding initializers in the SwiftProtobuf library, not generally
  /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
  /// initializers are defined in the SwiftProtobuf library. See the Message and
  /// Message+*Additions` files.
  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    _ = _uniqueStorage()
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      while let fieldNumber = try decoder.nextFieldNumber() {
        switch fieldNumber {
        case 1: try decoder.decodeSingularStringField(value: &_storage._name)
        case 2: try decoder.decodeSingularInt64Field(value: &_storage._execMicros)
        case 3: try decoder.decodeSingularInt64Field(value: &_storage._requestedBytes)
        case 4: try decoder.decodeSingularInt64Field(value: &_storage._parameters)
        case 6: try decoder.decodeSingularInt64Field(value: &_storage._totalExecMicros)
        case 7: try decoder.decodeSingularInt64Field(value: &_storage._totalRequestedBytes)
        case 8: try decoder.decodeSingularInt64Field(value: &_storage._totalParameters)
        case 10: try decoder.decodeRepeatedStringField(value: &_storage._devices)
        case 11: try decoder.decodeRepeatedMessageField(value: &_storage._shapes)
        case 12: try decoder.decodeRepeatedMessageField(value: &_storage._children)
        case 13: try decoder.decodeSingularInt64Field(value: &_storage._floatOps)
        case 14: try decoder.decodeSingularInt64Field(value: &_storage._totalFloatOps)
        case 15: try decoder.decodeSingularMessageField(value: &_storage._tensorValue)
        default: break
        }
      }
    }
  }

  /// Used by the encoding methods of the SwiftProtobuf library, not generally
  /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
  /// other serializer methods are defined in the SwiftProtobuf library. See the
  /// `Message` and `Message+*Additions` files.
  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    try withExtendedLifetime(_storage) { (_storage: _StorageClass) in
      if let v = _storage._name {
        try visitor.visitSingularStringField(value: v, fieldNumber: 1)
      }
      if let v = _storage._execMicros {
        try visitor.visitSingularInt64Field(value: v, fieldNumber: 2)
      }
      if let v = _storage._requestedBytes {
        try visitor.visitSingularInt64Field(value: v, fieldNumber: 3)
      }
      if let v = _storage._parameters {
        try visitor.visitSingularInt64Field(value: v, fieldNumber: 4)
      }
      if let v = _storage._totalExecMicros {
        try visitor.visitSingularInt64Field(value: v, fieldNumber: 6)
      }
      if let v = _storage._totalRequestedBytes {
        try visitor.visitSingularInt64Field(value: v, fieldNumber: 7)
      }
      if let v = _storage._totalParameters {
        try visitor.visitSingularInt64Field(value: v, fieldNumber: 8)
      }
      if !_storage._devices.isEmpty {
        try visitor.visitRepeatedStringField(value: _storage._devices, fieldNumber: 10)
      }
      if !_storage._shapes.isEmpty {
        try visitor.visitRepeatedMessageField(value: _storage._shapes, fieldNumber: 11)
      }
      if !_storage._children.isEmpty {
        try visitor.visitRepeatedMessageField(value: _storage._children, fieldNumber: 12)
      }
      if let v = _storage._floatOps {
        try visitor.visitSingularInt64Field(value: v, fieldNumber: 13)
      }
      if let v = _storage._totalFloatOps {
        try visitor.visitSingularInt64Field(value: v, fieldNumber: 14)
      }
      if let v = _storage._tensorValue {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 15)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  fileprivate var _storage = _StorageClass.defaultInstance
}

/// A node that groups multiple TFGraphNodeProto.
/// Depending on the 'view', the semantics of the TFmultiGraphNodeProto
/// is different:
/// code view: A node groups all TensorFlow graph nodes created by the
///            Python code.
/// op view:   A node groups all TensorFlow graph nodes that are of type
///            of the op (e.g. MatMul, Conv2D).
public struct Tensorflow_Tfprof_TFMultiGraphNodeProto: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".TFMultiGraphNodeProto"

  /// Name of the node.
  public var name: String {
    get {return _name ?? String()}
    set {_name = newValue}
  }
  /// Returns true if `name` has been explicitly set.
  public var hasName: Bool {return self._name != nil}
  /// Clears the value of `name`. Subsequent reads from it will return its default value.
  public mutating func clearName() {self._name = nil}

  /// code execution time.
  public var execMicros: Int64 {
    get {return _execMicros ?? 0}
    set {_execMicros = newValue}
  }
  /// Returns true if `execMicros` has been explicitly set.
  public var hasExecMicros: Bool {return self._execMicros != nil}
  /// Clears the value of `execMicros`. Subsequent reads from it will return its default value.
  public mutating func clearExecMicros() {self._execMicros = nil}

  /// Total requested bytes by the code.
  public var requestedBytes: Int64 {
    get {return _requestedBytes ?? 0}
    set {_requestedBytes = newValue}
  }
  /// Returns true if `requestedBytes` has been explicitly set.
  public var hasRequestedBytes: Bool {return self._requestedBytes != nil}
  /// Clears the value of `requestedBytes`. Subsequent reads from it will return its default value.
  public mutating func clearRequestedBytes() {self._requestedBytes = nil}

  /// Number of parameters if available.
  public var parameters: Int64 {
    get {return _parameters ?? 0}
    set {_parameters = newValue}
  }
  /// Returns true if `parameters` has been explicitly set.
  public var hasParameters: Bool {return self._parameters != nil}
  /// Clears the value of `parameters`. Subsequent reads from it will return its default value.
  public mutating func clearParameters() {self._parameters = nil}

  /// Number of float operations.
  public var floatOps: Int64 {
    get {return _floatOps ?? 0}
    set {_floatOps = newValue}
  }
  /// Returns true if `floatOps` has been explicitly set.
  public var hasFloatOps: Bool {return self._floatOps != nil}
  /// Clears the value of `floatOps`. Subsequent reads from it will return its default value.
  public mutating func clearFloatOps() {self._floatOps = nil}

  /// The following are the aggregated stats from descendants.
  /// The actual descendants depend on the data structure used.
  public var totalExecMicros: Int64 {
    get {return _totalExecMicros ?? 0}
    set {_totalExecMicros = newValue}
  }
  /// Returns true if `totalExecMicros` has been explicitly set.
  public var hasTotalExecMicros: Bool {return self._totalExecMicros != nil}
  /// Clears the value of `totalExecMicros`. Subsequent reads from it will return its default value.
  public mutating func clearTotalExecMicros() {self._totalExecMicros = nil}

  public var totalRequestedBytes: Int64 {
    get {return _totalRequestedBytes ?? 0}
    set {_totalRequestedBytes = newValue}
  }
  /// Returns true if `totalRequestedBytes` has been explicitly set.
  public var hasTotalRequestedBytes: Bool {return self._totalRequestedBytes != nil}
  /// Clears the value of `totalRequestedBytes`. Subsequent reads from it will return its default value.
  public mutating func clearTotalRequestedBytes() {self._totalRequestedBytes = nil}

  public var totalParameters: Int64 {
    get {return _totalParameters ?? 0}
    set {_totalParameters = newValue}
  }
  /// Returns true if `totalParameters` has been explicitly set.
  public var hasTotalParameters: Bool {return self._totalParameters != nil}
  /// Clears the value of `totalParameters`. Subsequent reads from it will return its default value.
  public mutating func clearTotalParameters() {self._totalParameters = nil}

  public var totalFloatOps: Int64 {
    get {return _totalFloatOps ?? 0}
    set {_totalFloatOps = newValue}
  }
  /// Returns true if `totalFloatOps` has been explicitly set.
  public var hasTotalFloatOps: Bool {return self._totalFloatOps != nil}
  /// Clears the value of `totalFloatOps`. Subsequent reads from it will return its default value.
  public mutating func clearTotalFloatOps() {self._totalFloatOps = nil}

  /// TensorFlow graph nodes contained by the TFMultiGraphNodeProto.
  public var graphNodes: [Tensorflow_Tfprof_TFGraphNodeProto] = []

  /// Descendants of the node. The actual descendants depend on the data
  /// structure used.
  public var children: [Tensorflow_Tfprof_TFMultiGraphNodeProto] = []

  public var unknownFields = SwiftProtobuf.UnknownStorage()

  public init() {}

  /// Used by the decoding initializers in the SwiftProtobuf library, not generally
  /// used directly. `init(serializedData:)`, `init(jsonUTF8Data:)`, and other decoding
  /// initializers are defined in the SwiftProtobuf library. See the Message and
  /// Message+*Additions` files.
  public mutating func decodeMessage<D: SwiftProtobuf.Decoder>(decoder: inout D) throws {
    while let fieldNumber = try decoder.nextFieldNumber() {
      switch fieldNumber {
      case 1: try decoder.decodeSingularStringField(value: &self._name)
      case 2: try decoder.decodeSingularInt64Field(value: &self._execMicros)
      case 3: try decoder.decodeSingularInt64Field(value: &self._requestedBytes)
      case 4: try decoder.decodeSingularInt64Field(value: &self._parameters)
      case 5: try decoder.decodeSingularInt64Field(value: &self._floatOps)
      case 6: try decoder.decodeSingularInt64Field(value: &self._totalExecMicros)
      case 7: try decoder.decodeSingularInt64Field(value: &self._totalRequestedBytes)
      case 8: try decoder.decodeSingularInt64Field(value: &self._totalParameters)
      case 9: try decoder.decodeSingularInt64Field(value: &self._totalFloatOps)
      case 10: try decoder.decodeRepeatedMessageField(value: &self.graphNodes)
      case 11: try decoder.decodeRepeatedMessageField(value: &self.children)
      default: break
      }
    }
  }

  /// Used by the encoding methods of the SwiftProtobuf library, not generally
  /// used directly. `Message.serializedData()`, `Message.jsonUTF8Data()`, and
  /// other serializer methods are defined in the SwiftProtobuf library. See the
  /// `Message` and `Message+*Additions` files.
  public func traverse<V: SwiftProtobuf.Visitor>(visitor: inout V) throws {
    if let v = self._name {
      try visitor.visitSingularStringField(value: v, fieldNumber: 1)
    }
    if let v = self._execMicros {
      try visitor.visitSingularInt64Field(value: v, fieldNumber: 2)
    }
    if let v = self._requestedBytes {
      try visitor.visitSingularInt64Field(value: v, fieldNumber: 3)
    }
    if let v = self._parameters {
      try visitor.visitSingularInt64Field(value: v, fieldNumber: 4)
    }
    if let v = self._floatOps {
      try visitor.visitSingularInt64Field(value: v, fieldNumber: 5)
    }
    if let v = self._totalExecMicros {
      try visitor.visitSingularInt64Field(value: v, fieldNumber: 6)
    }
    if let v = self._totalRequestedBytes {
      try visitor.visitSingularInt64Field(value: v, fieldNumber: 7)
    }
    if let v = self._totalParameters {
      try visitor.visitSingularInt64Field(value: v, fieldNumber: 8)
    }
    if let v = self._totalFloatOps {
      try visitor.visitSingularInt64Field(value: v, fieldNumber: 9)
    }
    if !self.graphNodes.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.graphNodes, fieldNumber: 10)
    }
    if !self.children.isEmpty {
      try visitor.visitRepeatedMessageField(value: self.children, fieldNumber: 11)
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  fileprivate var _name: String? = nil
  fileprivate var _execMicros: Int64? = nil
  fileprivate var _requestedBytes: Int64? = nil
  fileprivate var _parameters: Int64? = nil
  fileprivate var _floatOps: Int64? = nil
  fileprivate var _totalExecMicros: Int64? = nil
  fileprivate var _totalRequestedBytes: Int64? = nil
  fileprivate var _totalParameters: Int64? = nil
  fileprivate var _totalFloatOps: Int64? = nil
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow.tfprof"

extension Tensorflow_Tfprof_TFProfTensorProto: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "dtype"),
    2: .standard(proto: "value_double"),
    3: .standard(proto: "value_int64"),
    4: .standard(proto: "value_str"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Tfprof_TFProfTensorProto) -> Bool {
    if self._dtype != other._dtype {return false}
    if self.valueDouble != other.valueDouble {return false}
    if self.valueInt64 != other.valueInt64 {return false}
    if self.valueStr != other.valueStr {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_Tfprof_TFGraphNodeProto: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "name"),
    15: .standard(proto: "tensor_value"),
    2: .standard(proto: "exec_micros"),
    3: .standard(proto: "requested_bytes"),
    4: .same(proto: "parameters"),
    13: .standard(proto: "float_ops"),
    10: .same(proto: "devices"),
    6: .standard(proto: "total_exec_micros"),
    7: .standard(proto: "total_requested_bytes"),
    8: .standard(proto: "total_parameters"),
    14: .standard(proto: "total_float_ops"),
    11: .same(proto: "shapes"),
    12: .same(proto: "children"),
  ]

  fileprivate class _StorageClass {
    var _name: String? = nil
    var _tensorValue: Tensorflow_Tfprof_TFProfTensorProto? = nil
    var _execMicros: Int64? = nil
    var _requestedBytes: Int64? = nil
    var _parameters: Int64? = nil
    var _floatOps: Int64? = nil
    var _devices: [String] = []
    var _totalExecMicros: Int64? = nil
    var _totalRequestedBytes: Int64? = nil
    var _totalParameters: Int64? = nil
    var _totalFloatOps: Int64? = nil
    var _shapes: [Tensorflow_TensorShapeProto] = []
    var _children: [Tensorflow_Tfprof_TFGraphNodeProto] = []

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _name = source._name
      _tensorValue = source._tensorValue
      _execMicros = source._execMicros
      _requestedBytes = source._requestedBytes
      _parameters = source._parameters
      _floatOps = source._floatOps
      _devices = source._devices
      _totalExecMicros = source._totalExecMicros
      _totalRequestedBytes = source._totalRequestedBytes
      _totalParameters = source._totalParameters
      _totalFloatOps = source._totalFloatOps
      _shapes = source._shapes
      _children = source._children
    }
  }

  fileprivate mutating func _uniqueStorage() -> _StorageClass {
    if !isKnownUniquelyReferenced(&_storage) {
      _storage = _StorageClass(copying: _storage)
    }
    return _storage
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Tfprof_TFGraphNodeProto) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_storage, other_storage) in
        if _storage._name != other_storage._name {return false}
        if _storage._tensorValue != other_storage._tensorValue {return false}
        if _storage._execMicros != other_storage._execMicros {return false}
        if _storage._requestedBytes != other_storage._requestedBytes {return false}
        if _storage._parameters != other_storage._parameters {return false}
        if _storage._floatOps != other_storage._floatOps {return false}
        if _storage._devices != other_storage._devices {return false}
        if _storage._totalExecMicros != other_storage._totalExecMicros {return false}
        if _storage._totalRequestedBytes != other_storage._totalRequestedBytes {return false}
        if _storage._totalParameters != other_storage._totalParameters {return false}
        if _storage._totalFloatOps != other_storage._totalFloatOps {return false}
        if _storage._shapes != other_storage._shapes {return false}
        if _storage._children != other_storage._children {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_Tfprof_TFMultiGraphNodeProto: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .same(proto: "name"),
    2: .standard(proto: "exec_micros"),
    3: .standard(proto: "requested_bytes"),
    4: .same(proto: "parameters"),
    5: .standard(proto: "float_ops"),
    6: .standard(proto: "total_exec_micros"),
    7: .standard(proto: "total_requested_bytes"),
    8: .standard(proto: "total_parameters"),
    9: .standard(proto: "total_float_ops"),
    10: .standard(proto: "graph_nodes"),
    11: .same(proto: "children"),
  ]

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Tfprof_TFMultiGraphNodeProto) -> Bool {
    if self._name != other._name {return false}
    if self._execMicros != other._execMicros {return false}
    if self._requestedBytes != other._requestedBytes {return false}
    if self._parameters != other._parameters {return false}
    if self._floatOps != other._floatOps {return false}
    if self._totalExecMicros != other._totalExecMicros {return false}
    if self._totalRequestedBytes != other._totalRequestedBytes {return false}
    if self._totalParameters != other._totalParameters {return false}
    if self._totalFloatOps != other._totalFloatOps {return false}
    if self.graphNodes != other.graphNodes {return false}
    if self.children != other.children {return false}
    if unknownFields != other.unknownFields {return false}
    return true
  }
}