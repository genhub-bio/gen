use pyo3::{prelude::*, types::PyDict};

use crate::views::{
    block_group_viewer::PlotParameters,
    block_layout::{BaseLayout, ScaledLayout},
};

// BaseLayout class for visualization
#[pyclass]
pub struct PyBaseLayout {
    pub layout: BaseLayout,
}

#[pymethods]
impl PyBaseLayout {
    #[new]
    fn new(_py_graph: Bound<'_, PyDict>) -> PyResult<Self> {
        // In our main code, we build a BaseLayout from a graph object, not a blockgroup ID
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Direct instantiation from dictionary not yet implemented",
        ))
    }

    fn expand_right(&mut self) -> PyResult<()> {
        self.layout.expand_right();
        Ok(())
    }

    fn expand_left(&mut self) -> PyResult<()> {
        self.layout.expand_left();
        Ok(())
    }

    fn get_size(&self) -> PyResult<(f64, f64)> {
        Ok(self.layout.size)
    }

    fn create_scaled_layout(&self, label_width: u32, scale: u32) -> PyResult<PyScaledLayout> {
        let params = PlotParameters {
            label_width,
            scale,
            aspect_ratio: 1.0,
            edge_style: crate::views::block_group_viewer::EdgeStyle::Straight,
        };

        let scaled = ScaledLayout::from_base_layout(&self.layout, &params);

        Ok(PyScaledLayout {
            layout: scaled,
            parameters: params,
        })
    }
}

// ScaledLayout class for visualization
#[pyclass]
pub struct PyScaledLayout {
    pub layout: ScaledLayout,
    pub parameters: PlotParameters,
}

#[pymethods]
impl PyScaledLayout {
    fn refresh(&mut self, base_layout: &PyBaseLayout) -> PyResult<()> {
        self.layout.refresh(&base_layout.layout, &self.parameters);
        Ok(())
    }

    fn get_node_positions(&self, py: Python) -> PyResult<PyObject> {
        let result_dict = PyDict::new(py);

        for (node, pos) in self.layout.labels.iter() {
            let node_key = (
                node.block_id,
                node.node_id,
                node.sequence_start,
                node.sequence_end,
            );

            let position_value = (pos.0, pos.1);
            result_dict.set_item(node_key, position_value)?;
        }

        Ok(result_dict.into_pyobject(py)?.into())
    }

    fn get_edge_positions(&self, py: Python) -> PyResult<PyObject> {
        let result_dict = PyDict::new(py);

        for ((src, dst), pos) in self.layout.lines.iter() {
            let edge_key = (
                (
                    src.block_id,
                    src.node_id,
                    src.sequence_start,
                    src.sequence_end,
                ),
                (
                    dst.block_id,
                    dst.node_id,
                    dst.sequence_start,
                    dst.sequence_end,
                ),
            );

            let position_value = *pos;
            result_dict.set_item(edge_key, position_value)?;
        }

        Ok(result_dict.into_pyobject(py)?.into())
    }

    fn set_scale(&mut self, scale: u32) -> PyResult<()> {
        self.parameters.scale = scale;
        Ok(())
    }

    fn set_aspect_ratio(&mut self, aspect_ratio: f32) -> PyResult<()> {
        self.parameters.aspect_ratio = aspect_ratio;
        Ok(())
    }

    fn set_label_size(&mut self, width: u32, _height: u32) -> PyResult<()> {
        self.parameters.label_width = width;
        Ok(())
    }

    fn to_dict(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let result = PyDict::new(py);

            // Add basic layout parameters
            result.set_item("label_width", self.parameters.label_width)?;
            result.set_item("scale", self.parameters.scale)?;
            result.set_item("aspect_ratio", self.parameters.aspect_ratio)?;

            // Add node positions
            let nodes = PyDict::new(py);
            for (node, pos) in self.layout.labels.iter() {
                let node_key = format!(
                    "{}:{}:{}:{}",
                    node.block_id, node.node_id, node.sequence_start, node.sequence_end
                );
                let pos_value = (pos.0, pos.1); // ((x1, y1), (x2, y2))
                nodes.set_item(node_key, pos_value)?;
            }
            result.set_item("nodes", nodes)?;

            // Add edge positions
            let edges = PyDict::new(py);
            for ((src, dst), pos) in self.layout.lines.iter() {
                let edge_key = format!(
                    "{}:{}:{}:{}_{}:{}:{}:{}",
                    src.block_id,
                    src.node_id,
                    src.sequence_start,
                    src.sequence_end,
                    dst.block_id,
                    dst.node_id,
                    dst.sequence_start,
                    dst.sequence_end
                );
                let pos_value = (pos.0, pos.1); // ((x1, y1), (x2, y2))
                edges.set_item(edge_key, pos_value)?;
            }
            result.set_item("edges", edges)?;

            Ok(result.into_pyobject(py)?.into())
        })
    }
}
