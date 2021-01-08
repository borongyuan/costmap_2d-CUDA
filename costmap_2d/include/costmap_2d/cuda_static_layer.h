#include <costmap_2d/static_layer.h>
#include <tf2/convert.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#ifndef CUDA_STATIC_LAYER_H
#define CUDA_STATIC_LAYER_H
namespace costmap_2d
{
    namespace cuda
    {
        namespace static_layer
        {
            void rollingUpdateCosts(costmap_2d::Costmap2D& master_grid, tf2::Transform tf, costmap_2d::Costmap2D *staticLayer_costmap, costmap_2d::Costmap2D *layered_costmap, bool use_maximum, int min_x, int min_y, int max_x, int max_y);
        }
    }
}
#endif
