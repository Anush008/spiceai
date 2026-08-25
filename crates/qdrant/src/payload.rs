/*
Copyright 2024-2026 The Spice.ai OSS Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Result, UnsupportedPayloadTypeSnafu};
use arrow::array::Array;
use arrow_schema::DataType;
use qdrant_client::qdrant::value::Kind;
use qdrant_client::qdrant::{ListValue, PointId, PointStruct, Value};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct PointData {
    pub id: Option<PointId>,
    pub payload: HashMap<String, Value>,
    pub vector: Vec<f32>,
}

impl From<PointData> for PointStruct {
    fn from(point: PointData) -> Self {
        PointStruct {
            id: point.id,
            payload: point.payload,
            vectors: Some(point.vector.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: PointId,
    pub score: f32,
    pub payload: HashMap<String, Value>,
    pub vector: Option<Vec<f32>>,
}

impl From<qdrant_client::qdrant::ScoredPoint> for SearchResult {
    fn from(point: qdrant_client::qdrant::ScoredPoint) -> Self {
        let vector = point
            .vectors
            .as_ref()
            .and_then(qdrant_client::qdrant::VectorsOutput::get_vector)
            .and_then(|v| match v {
                qdrant_client::qdrant::vector_output::Vector::Dense(dense) => Some(dense.data),
                _ => None,
            });
        Self {
            id: point.id.unwrap_or_default(),
            score: point.score,
            payload: point.payload,
            vector,
        }
    }
}

/// Converts one Arrow cell into a Qdrant payload value.
///
/// # Errors
///
/// Returns an error for Arrow types Qdrant cannot store (and for unsigned
/// integers that exceed Qdrant's 64-bit integer range) rather than silently
/// storing NULL, which would corrupt the index.
pub fn arrow_value_to_qdrant(array: &dyn Array, row: usize) -> Result<Value> {
    use arrow::array::{
        BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int8Array, Int16Array,
        Int32Array, Int64Array, LargeStringArray, StringArray, StringViewArray, UInt8Array,
        UInt16Array, UInt32Array, UInt64Array,
    };
    fn unsupported(array: &dyn Array) -> crate::error::Error {
        UnsupportedPayloadTypeSnafu {
            arrow_type: format!("{:?}", array.data_type()),
        }
        .build()
    }
    macro_rules! payload {
        ($array:expr, $ty:ty, $row:expr, $conv:expr) => {
            $array
                .as_any()
                .downcast_ref::<$ty>()
                .map($conv)
                .ok_or_else(|| unsupported($array))
        };
    }
    if array.is_null(row) {
        return Ok(null_value());
    }
    match array.data_type() {
        DataType::Boolean => payload!(array, BooleanArray, row, |a| Value::from(a.value(row))),
        DataType::Int8 => {
            payload!(array, Int8Array, row, |a| Value::from(i64::from(
                a.value(row)
            )))
        }
        DataType::Int16 => {
            payload!(array, Int16Array, row, |a| Value::from(i64::from(
                a.value(row)
            )))
        }
        DataType::Int32 => {
            payload!(array, Int32Array, row, |a| Value::from(i64::from(
                a.value(row)
            )))
        }
        DataType::Int64 => payload!(array, Int64Array, row, |a| Value::from(a.value(row))),
        DataType::UInt8 => {
            payload!(array, UInt8Array, row, |a| Value::from(i64::from(
                a.value(row)
            )))
        }
        DataType::UInt16 => {
            payload!(array, UInt16Array, row, |a| Value::from(i64::from(
                a.value(row)
            )))
        }
        DataType::UInt32 => {
            payload!(array, UInt32Array, row, |a| Value::from(i64::from(
                a.value(row)
            )))
        }
        DataType::UInt64 => {
            // Qdrant integers are 64-bit signed; out-of-range values cannot be
            // stored faithfully, so fail instead of writing a NULL.
            let Some(uints) = array.as_any().downcast_ref::<UInt64Array>() else {
                return Err(unsupported(array));
            };
            i64::try_from(uints.value(row))
                .map(Value::from)
                .map_err(|_| unsupported(array))
        }
        DataType::FixedSizeList(_, size) => {
            let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() else {
                return Err(unsupported(array));
            };
            let size = usize::try_from(*size).unwrap_or(0);
            // `value_offset` accounts for zero-copy slices (`RecordBatch::slice`
            // shares buffers with a non-zero array offset); `row * size` alone
            // would read shifted values.
            let slot = usize::try_from(list.value_offset(row)).unwrap_or(0);
            let values = (0..size)
                .map(|i| arrow_value_to_qdrant(list.values().as_ref(), slot + i))
                .collect::<Result<Vec<_>>>()?;
            Ok(Value {
                kind: Some(Kind::ListValue(ListValue { values })),
            })
        }
        DataType::Float32 => payload!(array, Float32Array, row, |a| Value::from(a.value(row))),
        DataType::Float64 => payload!(array, Float64Array, row, |a| Value::from(a.value(row))),
        DataType::Utf8 => {
            payload!(array, StringArray, row, |a| Value::from(
                a.value(row).to_string()
            ))
        }
        DataType::LargeUtf8 => {
            payload!(array, LargeStringArray, row, |a| Value::from(
                a.value(row).to_string()
            ))
        }
        DataType::Utf8View => {
            payload!(array, StringViewArray, row, |a| Value::from(
                a.value(row).to_string()
            ))
        }
        _ => Err(unsupported(array)),
    }
}

fn null_value() -> Value {
    Value {
        kind: Some(Kind::NullValue(0)),
    }
}

#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub fn qdrant_value_to_arrow(value: &Value, data_type: &DataType) -> arrow::array::ArrayRef {
    use arrow::array::{
        ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int8Array,
        Int16Array, Int32Array, Int64Array, LargeStringArray, NullArray, StringArray,
        StringViewArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
    };
    match data_type {
        DataType::Boolean => Arc::new(BooleanArray::from(vec![match value.kind {
            Some(Kind::BoolValue(v)) => Some(v),
            _ => None,
        }])),
        DataType::Int8 => Arc::new(Int8Array::from(vec![match value.kind {
            Some(Kind::IntegerValue(v)) => i8::try_from(v).ok(),
            _ => None,
        }])),
        DataType::Int16 => Arc::new(Int16Array::from(vec![match value.kind {
            Some(Kind::IntegerValue(v)) => i16::try_from(v).ok(),
            _ => None,
        }])),
        DataType::Int32 => Arc::new(Int32Array::from(vec![match value.kind {
            Some(Kind::IntegerValue(v)) => i32::try_from(v).ok(),
            _ => None,
        }])),
        DataType::Int64 => Arc::new(Int64Array::from(vec![match value.kind {
            Some(Kind::IntegerValue(v)) => Some(v),
            _ => None,
        }])),
        DataType::UInt8 => Arc::new(UInt8Array::from(vec![match value.kind {
            Some(Kind::IntegerValue(v)) => u8::try_from(v).ok(),
            _ => None,
        }])),
        DataType::UInt16 => Arc::new(UInt16Array::from(vec![match value.kind {
            Some(Kind::IntegerValue(v)) => u16::try_from(v).ok(),
            _ => None,
        }])),
        DataType::UInt32 => Arc::new(UInt32Array::from(vec![match value.kind {
            Some(Kind::IntegerValue(v)) => u32::try_from(v).ok(),
            _ => None,
        }])),
        DataType::UInt64 => Arc::new(UInt64Array::from(vec![match value.kind {
            Some(Kind::IntegerValue(v)) => u64::try_from(v).ok(),
            _ => None,
        }])),
        DataType::Float32 => Arc::new(Float32Array::from(vec![match value.kind {
            Some(Kind::DoubleValue(f)) => Some(f as f32),
            Some(Kind::IntegerValue(v)) => Some(v as f32),
            _ => None,
        }])),
        DataType::Float64 => Arc::new(Float64Array::from(vec![match value.kind {
            Some(Kind::DoubleValue(f)) => Some(f),
            Some(Kind::IntegerValue(v)) => Some(v as f64),
            _ => None,
        }])),
        DataType::Utf8 => Arc::new(StringArray::from(vec![match &value.kind {
            Some(Kind::StringValue(v)) => Some(v.as_str()),
            _ => None,
        }])),
        DataType::LargeUtf8 => Arc::new(LargeStringArray::from(vec![match &value.kind {
            Some(Kind::StringValue(v)) => Some(v.clone()),
            _ => None,
        }])),
        DataType::Utf8View => Arc::new(StringViewArray::from(vec![match &value.kind {
            Some(Kind::StringValue(v)) => Some(v.clone()),
            _ => None,
        }])),
        DataType::FixedSizeList(field, size) => {
            let Some(Kind::ListValue(list)) = &value.kind else {
                tracing::warn!(
                    "Qdrant payload value is not a list where the dataset schema expects a fixed-size list; returning NULL"
                );
                return Arc::new(arrow::array::new_null_array(data_type, 1));
            };
            let child_cols: Vec<ArrayRef> = list
                .values
                .iter()
                .map(|v| qdrant_value_to_arrow(v, field.data_type()))
                .collect();
            let child: ArrayRef = if child_cols.is_empty() {
                arrow::array::new_null_array(field.data_type(), *size as usize)
            } else {
                let refs: Vec<&dyn Array> = child_cols.iter().map(AsRef::as_ref).collect();
                match arrow::compute::concat(&refs) {
                    Ok(combined) => combined,
                    Err(e) => {
                        tracing::warn!(
                            "Failed to combine Qdrant payload list values for the embedding column; returning NULL. Cause: {e}"
                        );
                        return Arc::new(arrow::array::new_null_array(data_type, 1));
                    }
                }
            };
            match FixedSizeListArray::try_new(Arc::clone(field), *size, child, None) {
                Ok(array) => Arc::new(array),
                Err(e) => {
                    tracing::warn!(
                        "Qdrant payload list does not match the dataset schema's fixed-size list type; returning NULL. Cause: {e}"
                    );
                    Arc::new(arrow::array::new_null_array(data_type, 1))
                }
            }
        }
        _ => Arc::new(NullArray::new(1)),
    }
}

#[must_use]
pub fn point_id_from_values(values: &[String]) -> PointId {
    let name = values.join("\u{1f}");
    PointId::from(Uuid::new_v5(&Uuid::NAMESPACE_URL, name.as_bytes()).to_string())
}
#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use arrow::array::{
        Float32Array, Int64Array, RecordBatch, StringArray, UInt8Array, UInt64Array,
    };

    use super::*;

    #[test]
    fn point_id_is_deterministic_and_order_sensitive() {
        let id1 = point_id_from_values(&["id-1".to_string()]);
        let id2 = point_id_from_values(&["id-1".to_string()]);
        let id3 = point_id_from_values(&["id-2".to_string()]);
        let composite = point_id_from_values(&["a".to_string(), "b".to_string()]);
        let composite_swapped = point_id_from_values(&["b".to_string(), "a".to_string()]);

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
        assert_ne!(composite, composite_swapped);
        assert!(matches!(
            id1.point_id_options,
            Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(_))
        ));
    }

    #[test]
    fn point_id_is_stable_uuid_v5() {
        let id = point_id_from_values(&["id-1".to_string()]);
        let Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) = id.point_id_options
        else {
            panic!("expected a UUID point id");
        };
        let expected = "6210c986-d7df-51f8-8e73-b4fc8e99b456";
        assert_eq!(uuid, expected);
        assert_eq!(
            uuid.parse::<uuid::Uuid>().expect("parse").get_version_num(),
            5
        );
    }

    #[test]
    fn u64_overflow_payload_is_rejected_not_corrupt() {
        let u64s = UInt64Array::from(vec![u64::MAX]);
        let err = arrow_value_to_qdrant(&u64s, 0).expect_err("out-of-range integer must error");
        assert!(err.to_string().contains("Unsupported Arrow type"));
    }

    #[test]
    fn unsupported_payload_type_is_rejected() {
        use arrow::array::Date32Array;
        let dates = Date32Array::from(vec![0]);
        let err = arrow_value_to_qdrant(&dates, 0).expect_err("unsupported type must error");
        assert!(err.to_string().contains("Date32"));
    }

    #[test]
    fn null_cell_stays_null_and_values_convert() {
        let strings = StringArray::from(vec![Some("x"), None]);
        let converted = arrow_value_to_qdrant(&strings, 1).expect("null cell");
        assert!(matches!(converted.kind, Some(Kind::NullValue(0))));
        let value = arrow_value_to_qdrant(&strings, 0).expect("string cell");
        assert_eq!(value.kind, Some(Kind::StringValue("x".to_string())));
    }

    #[test]
    fn sliced_fixed_size_list_converts_the_sliced_rows() {
        use arrow::array::{FixedSizeListArray, Int32Array};
        use arrow_schema::Field;
        use std::sync::Arc;

        let values = Int32Array::from(vec![1, 2, 3, 4]);
        let fsl = FixedSizeListArray::new(
            Arc::new(Field::new("item", arrow_schema::DataType::Int32, false)),
            2,
            Arc::new(values),
            None,
        );
        // Zero-copy slice: offset 1, length 1 — the second pair, not the first.
        let sliced = fsl.slice(1, 1);
        let converted = arrow_value_to_qdrant(&sliced, 0).expect("list payload");
        let Some(Kind::ListValue(list)) = converted.kind else {
            panic!("expected a list payload value");
        };
        let nums: Vec<i64> = list
            .values
            .iter()
            .map(|v| match &v.kind {
                Some(Kind::IntegerValue(n)) => *n,
                other => panic!("expected integer payload value, got {other:?}"),
            })
            .collect();
        assert_eq!(nums, vec![3, 4]);
    }

    #[test]
    fn arrow_payload_round_trips_through_qdrant_value() {
        let ints = Int64Array::from(vec![Some(42i64), None]);
        let strings = StringArray::from(vec![Some("hello"), Some("world")]);
        let batch_schema = arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("id", arrow_schema::DataType::Int64, true),
            arrow_schema::Field::new("name", arrow_schema::DataType::Utf8, true),
        ]);
        let batch = RecordBatch::try_new(
            Arc::new(batch_schema),
            vec![Arc::new(ints), Arc::new(strings)],
        )
        .expect("build batch");

        let mut payload = HashMap::new();
        payload.insert(
            "id".to_string(),
            arrow_value_to_qdrant(batch.column(0).as_ref(), 0).expect("int payload"),
        );
        payload.insert(
            "name".to_string(),
            arrow_value_to_qdrant(batch.column(1).as_ref(), 0).expect("string payload"),
        );

        let id_value = qdrant_value_to_arrow(
            payload.get("id").expect("id payload"),
            &arrow_schema::DataType::Int64,
        );
        let name_value = qdrant_value_to_arrow(
            payload.get("name").expect("name payload"),
            &arrow_schema::DataType::Utf8,
        );
        assert_eq!(
            id_value
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int64 array")
                .value(0),
            42
        );
        assert_eq!(
            name_value
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("string array")
                .value(0),
            "hello"
        );
    }

    #[test]
    fn narrow_int_payloads_null_propagate_on_out_of_range() {
        let too_big = Value::from(i64::MAX);
        let out = qdrant_value_to_arrow(&too_big, &arrow_schema::DataType::Int8);
        assert!(out.is_null(0));

        let fits = Value::from(7i64);
        let out = qdrant_value_to_arrow(&fits, &arrow_schema::DataType::Int8);
        assert!(!out.is_null(0));

        let unsigned = Value::from(7i64);
        let out = qdrant_value_to_arrow(&unsigned, &arrow_schema::DataType::UInt8);
        assert_eq!(
            out.as_any()
                .downcast_ref::<UInt8Array>()
                .expect("uint8 array")
                .value(0),
            7
        );

        let floating = Value::from(2.5f64);
        let out = qdrant_value_to_arrow(&floating, &arrow_schema::DataType::Float32);
        let value = out
            .as_any()
            .downcast_ref::<Float32Array>()
            .expect("float32 array")
            .value(0);
        assert!(
            (value - 2.5).abs() < f32::EPSILON,
            "expected 2.5, got {value}"
        );
    }
}
