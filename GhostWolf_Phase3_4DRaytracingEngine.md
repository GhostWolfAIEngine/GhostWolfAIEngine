# GhostWolf Voxelated Ray Inference Engine — Phase 3: 4D Raytracing Engine & Voxelizer Pipeline

## Overview
Phase 3 implements the core 4D raytracing engine and voxelizer pipeline that transforms the theoretical architecture from Phase 1 into a working inference system. This phase focuses on the mathematical foundations, data structures, and computational kernels that enable 120-way ray traversal across 4D voxel grids.

## Phase 3 Objectives
- **Primary Goal**: Implement working 4D raytracing engine with 120-ray inference
- **Secondary Goal**: Build voxelizer pipeline for model weight conversion
- **Tertiary Goal**: Establish memory management and scheduling infrastructure

## Core Components

### 1. 4D Voxel Grid Structure
```cpp
struct VoxelGrid4D {
    // Spatial dimensions (configurable, default 256³)
    uint32_t grid_size_x, grid_size_y, grid_size_z;
    
    // Time dimension slices (past/present/future)
    uint32_t time_slices;
    
    // Voxel data structure
    struct VoxelBrick {
        uint32_t brick_id;
        float features[FEATURE_CHANNELS];  // Embedding + attention features
        float time_weights[TIME_SLICES];   // Time-slice interpolation weights
        uint8_t compression_flags;         // Quantization metadata
    };
    
    // Sparse grid with spatial hashing
    std::unordered_map<uint64_t, VoxelBrick> sparse_grid;
    
    // Brick organization (32³ chunks for efficient traversal)
    static constexpr uint32_t BRICK_SIZE = 32;
    std::vector<VoxelBrick> brick_pool;
};
```

### 2. Ray Synthesis Engine
```cpp
struct Ray4D {
    // Spatial origin and direction
    float3 origin, direction;
    
    // Time perspective (past/present/future)
    float time_perspective;
    
    // Ray metadata
    uint32_t ray_id;
    float weight;
    RayGroup group;  // PAST, PRESENT, FUTURE
};

class RaySynthesizer {
private:
    // Deterministic low-discrepancy sequence for ray directions
    std::vector<float3> base_directions;
    
    // Ray grouping: 40 rays per time perspective
    static constexpr uint32_t RAYS_PER_GROUP = 40;
    static constexpr uint32_t TOTAL_RAYS = 120;
    
public:
    std::vector<Ray4D> synthesize_rays(const float3& context_center, 
                                      const float3& attention_focus);
    
    // Stratified sampling within each time group
    std::vector<Ray4D> sample_past_rays(const float3& center, float time_offset);
    std::vector<Ray4D> sample_present_rays(const float3& center);
    std::vector<Ray4D> sample_future_rays(const float3& center, float time_offset);
};
```

### 3. Voxelizer Pipeline
```cpp
class Voxelizer {
private:
    // Projection matrices for embedding → 3D coordinates
    std::vector<Eigen::MatrixXf> projection_matrices;
    
    // Quantization parameters
    struct QuantizationConfig {
        float scale_factor;
        float zero_point;
        uint8_t bit_depth;  // 8-bit or 16-bit
    };

public:
    // Main voxelization pipeline
    VoxelGrid4D voxelize_model(const ModelWeights& weights, 
                               const VoxelizationConfig& config);
    
    // Embedding projection to 3D coordinates
    float3 project_embedding(const std::vector<float>& embedding, int layer_id);
    
    // Attention weight encoding into voxel features
    void encode_attention_weights(VoxelGrid4D& grid, 
                                 const AttentionWeights& attn_weights,
                                 int layer_id);
    
    // Time dimension initialization
    void initialize_time_slices(VoxelGrid4D& grid, 
                               const std::vector<float>& temporal_weights);
};
```

### 4. Ray Traversal Engine
```cpp
class RayTraverser {
private:
    // Traversal state
    struct TraversalState {
        float3 current_pos;
        float current_time;
        std::vector<float> accumulated_features;
        float traversal_distance;
    };
    
    // Spatial acceleration structures
    std::vector<BoundingBox> brick_bounds;
    std::vector<uint32_t> spatial_index;

public:
    // Main traversal function
    TraversalResult traverse_ray(const Ray4D& ray, const VoxelGrid4D& grid);
    
    // Brick intersection testing
    std::vector<uint32_t> find_intersecting_bricks(const Ray4D& ray, 
                                                  const VoxelGrid4D& grid);
    
    // Feature sampling with trilinear interpolation
    float sample_voxel_features(const float3& sample_pos, 
                               const VoxelBrick& brick);
    
    // Time-slice interpolation
    float interpolate_time_slice(const float time_pos, 
                                const std::vector<float>& time_weights);
};
```

### 5. Feature Aggregation & Collapse
```cpp
class FeatureAggregator {
private:
    // Per-ray feature accumulation
    struct RayFeatures {
        uint32_t ray_id;
        std::vector<float> features;
        float confidence;
    };

public:
    // Aggregate features across all 120 rays
    std::vector<float> aggregate_ray_features(const std::vector<RayFeatures>& ray_features);
    
    // Mean collapse with group weighting
    std::vector<float> collapse_to_logits(const std::vector<RayFeatures>& ray_features);
    
    // Group-specific weighting (past/present/future)
    float calculate_group_weight(RayGroup group);
    
    // Confidence-based feature selection
    std::vector<float> select_high_confidence_features(const std::vector<RayFeatures>& features,
                                                      float confidence_threshold);
};
```

## Mathematical Foundations

### 1. 4D Ray-Volume Intersection
```math
\text{Ray equation: } \mathbf{r}(t) = \mathbf{o} + t\mathbf{d}
\text{ where } \mathbf{o} = (x_0, y_0, z_0, t_0), \mathbf{d} = (d_x, d_y, d_z, d_t)

\text{Voxel sampling: } \mathbf{v}(i,j,k,l) = \mathbf{v}_0 + i\Delta x + j\Delta y + k\Delta z + l\Delta t

\text{Intersection test: } \text{Find } t \text{ such that } \mathbf{r}(t) \in \text{voxel grid}
```

### 2. Feature Interpolation
```math
\text{Trilinear interpolation in space:}
f(x,y,z) = \sum_{i,j,k \in \{0,1\}} f_{ijk} \cdot w_i(x) \cdot w_j(y) \cdot w_k(z)

\text{Linear interpolation in time:}
f(t) = f_0 \cdot (1-\alpha) + f_1 \cdot \alpha \text{ where } \alpha = \frac{t-t_0}{t_1-t_0}
```

### 3. Ray Weighting & Collapse
```math
\text{Group weights: } w_{\text{past}} = w_{\text{present}} = w_{\text{future}} = \frac{1}{3}

\text{Final logits: } \mathbf{L} = \frac{1}{120} \sum_{i=1}^{120} w_i \cdot \mathbf{f}_i

\text{where } w_i = \frac{1}{40} \text{ within each group}
```

## Implementation Phases

### Phase 3.1: Core Data Structures (Week 1-2)
- [ ] Implement `VoxelGrid4D` class with sparse grid support
- [ ] Build `VoxelBrick` structure with feature encoding
- [ ] Create spatial hashing for efficient voxel lookup
- [ ] Implement brick pooling and memory management

### Phase 3.2: Ray Synthesis (Week 3-4)
- [ ] Implement `RaySynthesizer` with deterministic sampling
- [ ] Build 120-ray generation system (40 per time group)
- [ ] Create stratified sampling within each group
- [ ] Add ray weighting and metadata

### Phase 3.3: Voxelizer Pipeline (Week 5-6)
- [ ] Implement embedding projection to 3D coordinates
- [ ] Build attention weight encoding into voxel features
- [ ] Create time-slice initialization system
- [ ] Add quantization and compression support

### Phase 3.4: Ray Traversal (Week 7-8)
- [ ] Implement `RayTraverser` with brick intersection testing
- [ ] Build trilinear interpolation for spatial sampling
- [ ] Create time-slice interpolation system
- [ ] Add acceleration structures for efficient traversal

### Phase 3.5: Feature Aggregation (Week 9-10)
- [ ] Implement per-ray feature accumulation
- [ ] Build 120-ray aggregation system
- [ ] Create mean collapse with group weighting
- [ ] Add confidence-based feature selection

### Phase 3.6: Integration & Testing (Week 11-12)
- [ ] Integrate all components into unified pipeline
- [ ] Implement end-to-end inference loop
- [ ] Add performance profiling and optimization
- [ ] Create comprehensive test suite

## Performance Targets

### Memory Efficiency
- **Voxel Grid**: ≤ 8GB for 7B parameter models
- **Ray Buffers**: ≤ 2GB for 120-ray batches
- **Feature Cache**: ≤ 1GB for hot voxel regions

### Computational Performance
- **Ray Synthesis**: ≤ 1ms for 120 rays
- **Voxel Traversal**: ≤ 5ms per ray (600ms total for 120 rays)
- **Feature Aggregation**: ≤ 2ms for 120-ray collapse
- **Total Latency**: ≤ 1 second per token (target: 10-20 tokens/second)

### Scalability
- **Grid Resolution**: Configurable from 128³ to 512³
- **Time Slices**: Configurable from 8 to 32 slices
- **Ray Count**: Configurable from 60 to 240 rays

## Testing & Validation

### Unit Tests
- Voxel grid operations (insertion, lookup, deletion)
- Ray synthesis (direction sampling, grouping)
- Voxelization (embedding projection, feature encoding)
- Traversal (intersection testing, interpolation)
- Aggregation (feature accumulation, collapse)

### Integration Tests
- End-to-end voxelization pipeline
- Complete ray traversal system
- Full 120-ray inference loop
- Memory management and cleanup

### Performance Tests
- Memory usage profiling
- Computational latency measurement
- Scalability testing with different grid sizes
- Ray count scaling analysis

## Dependencies & Integration

### External Libraries
- **Eigen3**: Matrix operations and linear algebra
- **DirectX 12**: GPU compute for voxel math
- **DirectML**: NPU operations for prefetch
- **ONNX Runtime**: Model weight loading

### Internal Dependencies
- **Phase 1**: Architecture and design patterns
- **Phase 2**: Model adapters and weight loading
- **Future Phases**: GUI, scheduling, and optimization

## Risk Mitigation

### Technical Risks
- **NPU API Variability**: Fallback to GPU compute for unsupported operations
- **Memory Pressure**: Implement aggressive LRU eviction and streaming loads
- **Numerical Stability**: Use double precision for critical calculations, validate with reference implementations

### Performance Risks
- **Ray Traversal Bottleneck**: Implement spatial acceleration structures and early termination
- **Memory Bandwidth**: Use pinned memory and DMA transfers, implement double/triple buffering
- **Load Balancing**: Dynamic work distribution between CPU/GPU/NPU

## Success Criteria

### Functional Requirements
- [ ] Successfully voxelize 7B parameter models
- [ ] Generate and traverse 120 rays per inference step
- [ ] Produce coherent token outputs from ray aggregation
- [ ] Maintain memory usage within target limits

### Performance Requirements
- [ ] Achieve ≤ 1 second latency per token
- [ ] Support models up to 7B parameters
- [ ] Maintain stable memory usage under load
- [ ] Provide configurable quality/speed tradeoffs

### Quality Requirements
- [ ] Pass comprehensive test suite
- [ ] Maintain numerical stability across operations
- [ ] Provide detailed logging and debugging
- [ ] Support graceful error handling and recovery

## Next Phase Preparation

Phase 3 establishes the computational foundation for the GhostWolf engine. The next phase (Phase 4) will focus on:

1. **Scheduler & Runtime**: Job graph execution and device coordination
2. **Memory Manager**: Advanced memory pooling and optimization
3. **Performance Optimization**: GPU/NPU kernel optimization and load balancing
4. **GUI Integration**: Qt interface for real-time monitoring and control

This phase represents the core algorithmic implementation that transforms the theoretical architecture into a working inference system.
