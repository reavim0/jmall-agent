# Task: generic accessories query schema

Date: 2026-04-28

## User Requirements

- `shopping_context.targets[].category` uses `accessories` for phone accessories.
- Canonical category values are schema enum values. Current values: `cell_phone`, `accessories`.
- Do not add code-level category alias maps; constrain model output with the field schema and validate against the enum.
- `category` is required in `query_plan`, but inherited from `shopping_context`; the model should not invent it.
- `query_plan.lexical_query.must.sub_category` is required for `accessories`.
- `query_plan.lexical_query.must.compatibility` is required for `accessories`.
- `compatibility` defaults to `null`, meaning all compatible / no compatibility restriction.
- If context identifies a required compatible model, connector, device, or platform, the model must fill `compatibility` with that value.
- `sub_category` is shared by semantic and keyword retrieval intent.
- Allowed accessory `sub_category` values are fixed for now; if the item does not belong to them, use `others`.
- Do not create separate detailed schemas for phone case, film, charger, power bank, cable, headset, etc.
- Keep the accessories schema generic: use `preferences`, `dislikes`, and `others` instead of per-accessory fine-grained fields.

## Current Implementation Contract

- `category_query_schema.category == "accessories"`.
- `lexical_fields.must == ["category", "sub_category", "compatibility"]`.
- `lexical_fields.should == ["brand", "preferences", "others"]`.
- `lexical_fields.must_not == ["sub_category", "dislikes", "seller_risk", "condition", "others"]`.
- `sub_category_enum == ["耳机", "手机壳", "膜", "充电器", "充电宝", "数据线", "支架", "转接头", "others"]`.

## Notes

- Product catalog/search support for accessories is separate work; this task only fixes the agent/query schema side.
- Do not reintroduce per-accessory query schemas unless the user explicitly changes direction.
