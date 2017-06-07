// DO NOT EDIT.
//
// Generated by the Swift generator plugin for the protocol buffer compiler.
// Source: tensorflow_serving/servables/tensorflow/session_bundle_config.proto
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

/// Configuration parameters for a SessionBundle, with optional batching.
public struct Tensorflow_Serving_SessionBundleConfig: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".SessionBundleConfig"

  /// The TensorFlow runtime to connect to.
  /// See full documentation in tensorflow/core/public/session_options.h.
  ///
  /// For single machine serving, we recommend using the empty string "", which
  /// will configure the local TensorFlow runtime implementation. This provides
  /// the best isolation currently available across multiple Session servables.
  public var sessionTarget: String {
    get {return _storage._sessionTarget}
    set {_uniqueStorage()._sessionTarget = newValue}
  }

  /// TensorFlow Session configuration options.
  /// See details at tensorflow/core/protobuf/config.proto.
  public var sessionConfig: Tensorflow_ConfigProto {
    get {return _storage._sessionConfig ?? Tensorflow_ConfigProto()}
    set {_uniqueStorage()._sessionConfig = newValue}
  }
  /// Returns true if `sessionConfig` has been explicitly set.
  public var hasSessionConfig: Bool {return _storage._sessionConfig != nil}
  /// Clears the value of `sessionConfig`. Subsequent reads from it will return its default value.
  public mutating func clearSessionConfig() {_storage._sessionConfig = nil}

  /// If set, each emitted session is wrapped with a layer that schedules Run()
  /// calls in batches. The batching layer is transparent to the client
  /// (implements the tensorflow::Session API).
  ///
  /// IMPORTANT: With batching enabled, client threads will spend most of their
  /// time blocked on Session::Run() calls, waiting for enough peer threads to
  /// also call Session::Run() such that a large batch can be formed. For good
  /// throughput, we recommend setting the number of client threads equal to
  /// roughly twice the maximum batch size ('max_batch_size' below).
  ///
  /// The batching layer uses a SharedBatchScheduler to coordinate batching
  /// across multiple session servables emitted by this source adapter. A
  /// BatchSchedulerRetrier is added on top of each batching session.
  public var batchingParameters: Tensorflow_Serving_BatchingParameters {
    get {return _storage._batchingParameters ?? Tensorflow_Serving_BatchingParameters()}
    set {_uniqueStorage()._batchingParameters = newValue}
  }
  /// Returns true if `batchingParameters` has been explicitly set.
  public var hasBatchingParameters: Bool {return _storage._batchingParameters != nil}
  /// Clears the value of `batchingParameters`. Subsequent reads from it will return its default value.
  public mutating func clearBatchingParameters() {_storage._batchingParameters = nil}

  /// If set, session run calls use a separate threadpool for restore and init
  /// ops as part of loading the session-bundle. The value of this field should
  /// correspond to the index of the tensorflow::ThreadPoolOptionProto defined as
  /// part of `session_config.session_inter_op_thread_pool`.
  public var sessionRunLoadThreadpoolIndex: SwiftProtobuf.Google_Protobuf_Int32Value {
    get {return _storage._sessionRunLoadThreadpoolIndex ?? SwiftProtobuf.Google_Protobuf_Int32Value()}
    set {_uniqueStorage()._sessionRunLoadThreadpoolIndex = newValue}
  }
  /// Returns true if `sessionRunLoadThreadpoolIndex` has been explicitly set.
  public var hasSessionRunLoadThreadpoolIndex: Bool {return _storage._sessionRunLoadThreadpoolIndex != nil}
  /// Clears the value of `sessionRunLoadThreadpoolIndex`. Subsequent reads from it will return its default value.
  public mutating func clearSessionRunLoadThreadpoolIndex() {_storage._sessionRunLoadThreadpoolIndex = nil}

  /// EXPERIMENTAL. THIS FIELD MAY CHANGE OR GO AWAY. USE WITH CAUTION.
  ///
  /// Input tensors to append to every Session::Run() call.
  public var experimentalFixedInputTensors: [Tensorflow_NamedTensorProto] {
    get {return _storage._experimentalFixedInputTensors}
    set {_uniqueStorage()._experimentalFixedInputTensors = newValue}
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
        case 1: try decoder.decodeSingularStringField(value: &_storage._sessionTarget)
        case 2: try decoder.decodeSingularMessageField(value: &_storage._sessionConfig)
        case 3: try decoder.decodeSingularMessageField(value: &_storage._batchingParameters)
        case 4: try decoder.decodeSingularMessageField(value: &_storage._sessionRunLoadThreadpoolIndex)
        case 778: try decoder.decodeRepeatedMessageField(value: &_storage._experimentalFixedInputTensors)
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
      if !_storage._sessionTarget.isEmpty {
        try visitor.visitSingularStringField(value: _storage._sessionTarget, fieldNumber: 1)
      }
      if let v = _storage._sessionConfig {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
      }
      if let v = _storage._batchingParameters {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 3)
      }
      if let v = _storage._sessionRunLoadThreadpoolIndex {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 4)
      }
      if !_storage._experimentalFixedInputTensors.isEmpty {
        try visitor.visitRepeatedMessageField(value: _storage._experimentalFixedInputTensors, fieldNumber: 778)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  fileprivate var _storage = _StorageClass.defaultInstance
}

/// Batching parameters. Each individual parameter is optional. If omitted, the
/// default value from the relevant batching config struct (SharedBatchScheduler
/// ::Options or BatchSchedulerRetrier::Options) is used.
public struct Tensorflow_Serving_BatchingParameters: SwiftProtobuf.Message {
  public static let protoMessageName: String = _protobuf_package + ".BatchingParameters"

  /// The maximum size of each batch.
  ///
  /// IMPORTANT: As discussed above, use 'max_batch_size * 2' client threads to
  /// achieve high throughput with batching.
  public var maxBatchSize: SwiftProtobuf.Google_Protobuf_Int64Value {
    get {return _storage._maxBatchSize ?? SwiftProtobuf.Google_Protobuf_Int64Value()}
    set {_uniqueStorage()._maxBatchSize = newValue}
  }
  /// Returns true if `maxBatchSize` has been explicitly set.
  public var hasMaxBatchSize: Bool {return _storage._maxBatchSize != nil}
  /// Clears the value of `maxBatchSize`. Subsequent reads from it will return its default value.
  public mutating func clearMaxBatchSize() {_storage._maxBatchSize = nil}

  /// If a task has been enqueued for this amount of time (in microseconds), and
  /// a thread is available, the scheduler will immediately form a batch from
  /// enqueued tasks and assign the batch to the thread for processing, even if
  /// the batch's size is below 'max_batch_size'.
  public var batchTimeoutMicros: SwiftProtobuf.Google_Protobuf_Int64Value {
    get {return _storage._batchTimeoutMicros ?? SwiftProtobuf.Google_Protobuf_Int64Value()}
    set {_uniqueStorage()._batchTimeoutMicros = newValue}
  }
  /// Returns true if `batchTimeoutMicros` has been explicitly set.
  public var hasBatchTimeoutMicros: Bool {return _storage._batchTimeoutMicros != nil}
  /// Clears the value of `batchTimeoutMicros`. Subsequent reads from it will return its default value.
  public mutating func clearBatchTimeoutMicros() {_storage._batchTimeoutMicros = nil}

  /// The maximum length of the queue, in terms of the number of batches. (A
  /// batch that has been scheduled on a thread is considered to have been
  /// removed from the queue.)
  public var maxEnqueuedBatches: SwiftProtobuf.Google_Protobuf_Int64Value {
    get {return _storage._maxEnqueuedBatches ?? SwiftProtobuf.Google_Protobuf_Int64Value()}
    set {_uniqueStorage()._maxEnqueuedBatches = newValue}
  }
  /// Returns true if `maxEnqueuedBatches` has been explicitly set.
  public var hasMaxEnqueuedBatches: Bool {return _storage._maxEnqueuedBatches != nil}
  /// Clears the value of `maxEnqueuedBatches`. Subsequent reads from it will return its default value.
  public mutating func clearMaxEnqueuedBatches() {_storage._maxEnqueuedBatches = nil}

  /// The number of threads to use to process batches.
  /// Must be >= 1, and should be tuned carefully.
  public var numBatchThreads: SwiftProtobuf.Google_Protobuf_Int64Value {
    get {return _storage._numBatchThreads ?? SwiftProtobuf.Google_Protobuf_Int64Value()}
    set {_uniqueStorage()._numBatchThreads = newValue}
  }
  /// Returns true if `numBatchThreads` has been explicitly set.
  public var hasNumBatchThreads: Bool {return _storage._numBatchThreads != nil}
  /// Clears the value of `numBatchThreads`. Subsequent reads from it will return its default value.
  public mutating func clearNumBatchThreads() {_storage._numBatchThreads = nil}

  /// The name to use for the pool of batch threads.
  public var threadPoolName: SwiftProtobuf.Google_Protobuf_StringValue {
    get {return _storage._threadPoolName ?? SwiftProtobuf.Google_Protobuf_StringValue()}
    set {_uniqueStorage()._threadPoolName = newValue}
  }
  /// Returns true if `threadPoolName` has been explicitly set.
  public var hasThreadPoolName: Bool {return _storage._threadPoolName != nil}
  /// Clears the value of `threadPoolName`. Subsequent reads from it will return its default value.
  public mutating func clearThreadPoolName() {_storage._threadPoolName = nil}

  /// The allowed batch sizes. (Ignored if left empty.)
  /// Requirements:
  ///  - The entries must be in increasing order.
  ///  - The final entry must equal 'max_batch_size'.
  public var allowedBatchSizes: [Int64] {
    get {return _storage._allowedBatchSizes}
    set {_uniqueStorage()._allowedBatchSizes = newValue}
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
        case 1: try decoder.decodeSingularMessageField(value: &_storage._maxBatchSize)
        case 2: try decoder.decodeSingularMessageField(value: &_storage._batchTimeoutMicros)
        case 3: try decoder.decodeSingularMessageField(value: &_storage._maxEnqueuedBatches)
        case 4: try decoder.decodeSingularMessageField(value: &_storage._numBatchThreads)
        case 5: try decoder.decodeSingularMessageField(value: &_storage._threadPoolName)
        case 6: try decoder.decodeRepeatedInt64Field(value: &_storage._allowedBatchSizes)
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
      if let v = _storage._maxBatchSize {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 1)
      }
      if let v = _storage._batchTimeoutMicros {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 2)
      }
      if let v = _storage._maxEnqueuedBatches {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 3)
      }
      if let v = _storage._numBatchThreads {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 4)
      }
      if let v = _storage._threadPoolName {
        try visitor.visitSingularMessageField(value: v, fieldNumber: 5)
      }
      if !_storage._allowedBatchSizes.isEmpty {
        try visitor.visitPackedInt64Field(value: _storage._allowedBatchSizes, fieldNumber: 6)
      }
    }
    try unknownFields.traverse(visitor: &visitor)
  }

  fileprivate var _storage = _StorageClass.defaultInstance
}

// MARK: - Code below here is support for the SwiftProtobuf runtime.

fileprivate let _protobuf_package = "tensorflow.serving"

extension Tensorflow_Serving_SessionBundleConfig: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "session_target"),
    2: .standard(proto: "session_config"),
    3: .standard(proto: "batching_parameters"),
    4: .standard(proto: "session_run_load_threadpool_index"),
    778: .standard(proto: "experimental_fixed_input_tensors"),
  ]

  fileprivate class _StorageClass {
    var _sessionTarget: String = String()
    var _sessionConfig: Tensorflow_ConfigProto? = nil
    var _batchingParameters: Tensorflow_Serving_BatchingParameters? = nil
    var _sessionRunLoadThreadpoolIndex: SwiftProtobuf.Google_Protobuf_Int32Value? = nil
    var _experimentalFixedInputTensors: [Tensorflow_NamedTensorProto] = []

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _sessionTarget = source._sessionTarget
      _sessionConfig = source._sessionConfig
      _batchingParameters = source._batchingParameters
      _sessionRunLoadThreadpoolIndex = source._sessionRunLoadThreadpoolIndex
      _experimentalFixedInputTensors = source._experimentalFixedInputTensors
    }
  }

  fileprivate mutating func _uniqueStorage() -> _StorageClass {
    if !isKnownUniquelyReferenced(&_storage) {
      _storage = _StorageClass(copying: _storage)
    }
    return _storage
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Serving_SessionBundleConfig) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_storage, other_storage) in
        if _storage._sessionTarget != other_storage._sessionTarget {return false}
        if _storage._sessionConfig != other_storage._sessionConfig {return false}
        if _storage._batchingParameters != other_storage._batchingParameters {return false}
        if _storage._sessionRunLoadThreadpoolIndex != other_storage._sessionRunLoadThreadpoolIndex {return false}
        if _storage._experimentalFixedInputTensors != other_storage._experimentalFixedInputTensors {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}

extension Tensorflow_Serving_BatchingParameters: SwiftProtobuf._MessageImplementationBase, SwiftProtobuf._ProtoNameProviding {
  public static let _protobuf_nameMap: SwiftProtobuf._NameMap = [
    1: .standard(proto: "max_batch_size"),
    2: .standard(proto: "batch_timeout_micros"),
    3: .standard(proto: "max_enqueued_batches"),
    4: .standard(proto: "num_batch_threads"),
    5: .standard(proto: "thread_pool_name"),
    6: .standard(proto: "allowed_batch_sizes"),
  ]

  fileprivate class _StorageClass {
    var _maxBatchSize: SwiftProtobuf.Google_Protobuf_Int64Value? = nil
    var _batchTimeoutMicros: SwiftProtobuf.Google_Protobuf_Int64Value? = nil
    var _maxEnqueuedBatches: SwiftProtobuf.Google_Protobuf_Int64Value? = nil
    var _numBatchThreads: SwiftProtobuf.Google_Protobuf_Int64Value? = nil
    var _threadPoolName: SwiftProtobuf.Google_Protobuf_StringValue? = nil
    var _allowedBatchSizes: [Int64] = []

    static let defaultInstance = _StorageClass()

    private init() {}

    init(copying source: _StorageClass) {
      _maxBatchSize = source._maxBatchSize
      _batchTimeoutMicros = source._batchTimeoutMicros
      _maxEnqueuedBatches = source._maxEnqueuedBatches
      _numBatchThreads = source._numBatchThreads
      _threadPoolName = source._threadPoolName
      _allowedBatchSizes = source._allowedBatchSizes
    }
  }

  fileprivate mutating func _uniqueStorage() -> _StorageClass {
    if !isKnownUniquelyReferenced(&_storage) {
      _storage = _StorageClass(copying: _storage)
    }
    return _storage
  }

  public func _protobuf_generated_isEqualTo(other: Tensorflow_Serving_BatchingParameters) -> Bool {
    if _storage !== other._storage {
      let storagesAreEqual: Bool = withExtendedLifetime((_storage, other._storage)) { (_storage, other_storage) in
        if _storage._maxBatchSize != other_storage._maxBatchSize {return false}
        if _storage._batchTimeoutMicros != other_storage._batchTimeoutMicros {return false}
        if _storage._maxEnqueuedBatches != other_storage._maxEnqueuedBatches {return false}
        if _storage._numBatchThreads != other_storage._numBatchThreads {return false}
        if _storage._threadPoolName != other_storage._threadPoolName {return false}
        if _storage._allowedBatchSizes != other_storage._allowedBatchSizes {return false}
        return true
      }
      if !storagesAreEqual {return false}
    }
    if unknownFields != other.unknownFields {return false}
    return true
  }
}