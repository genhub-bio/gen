//! Derive support for typed, fluent SQL selectors in `gen-models`.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::{
    Attribute, Data, DeriveInput, Error, ExprPath, Fields, Ident, LitBool, LitStr, Type,
    parse_macro_input,
};

/// Generates a `<Model>Select` query builder from a persisted model's named fields.
///
/// String fields receive exact and `_contains` filters. Every generated field can be used for
/// typed ordering, projections, and explicit join conditions. Generated selectors can be composed
/// with `.join_on(left_field, right_field)` or `.join_filtered_on(...)`. Joined queries can project
/// fields with `.only(...)` or complete models with `.models::<(...)>()`.
/// Primary-key selectors also provide `get_by_id(...)` and ordered, deduplicated
/// `query_by_ids(...)` loads.
/// `#[model_select(table = "...")]` generates the model's `Query` implementation.
/// `#[model_select(column = "...")]` overrides a field's SQL column,
/// `#[model_select(primary_key)]` marks a non-`id` primary key, `#[model_select(skip)]` excludes a
/// field, and the struct-level `history`, `from_row`, `source`, `alias`, and `select` options
/// support custom persistence behavior or aliased queries.
#[proc_macro_derive(ModelSelect, attributes(model_select))]
pub fn derive_model_select(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_model_select(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn expand_model_select(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let model = input.ident;
    let fields = match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => fields.named,
            _ => {
                return Err(Error::new_spanned(
                    model,
                    "ModelSelect can only be derived for structs with named fields",
                ));
            }
        },
        _ => {
            return Err(Error::new_spanned(
                model,
                "ModelSelect can only be derived for structs",
            ));
        }
    };

    if !input.generics.params.is_empty() {
        return Err(Error::new_spanned(
            input.generics,
            "ModelSelect does not support generic model structs",
        ));
    }

    let options = ContainerOptions::from_attributes(&input.attrs)?;
    let selector = format_ident!("{}Select", model);
    let source_alias = options
        .alias
        .as_ref()
        .map(|alias| {
            let alias = LitStr::new(alias, Span::call_site());
            quote! { #alias }
        })
        .unwrap_or_else(|| {
            quote! { <#model as ::gen_models::traits::Query>::TABLE_NAME }
        });

    let mut variants = Vec::new();
    let mut column_constants = Vec::new();
    let mut filter_methods = Vec::new();
    let mut model_columns = Vec::new();
    let mut model_initializers = Vec::new();
    let mut query_initializers = Vec::new();
    let mut has_skipped_field = false;
    let mut explicit_primary_key = None;
    let mut inferred_primary_key = None;

    for field in fields {
        let Some(field_name) = field.ident else {
            continue;
        };
        let field_options = FieldOptions::from_attributes(&field.attrs)?;
        if field_options.skip {
            if field_options.primary_key {
                return Err(Error::new_spanned(
                    field_name,
                    "a ModelSelect primary key cannot be skipped",
                ));
            }
            has_skipped_field = true;
            continue;
        }

        let variant = snake_to_pascal(&field_name);
        let column = field_options
            .column
            .unwrap_or_else(|| field_name_string(&field_name));
        let column_literal = LitStr::new(&column, Span::call_site());
        let field_type = &field.ty;
        let value_type = option_inner(field_type).unwrap_or(field_type);
        let field_index = model_initializers.len();

        let primary_key = (
            field_name.clone(),
            value_type.clone(),
            is_string(value_type),
            column_literal.clone(),
        );
        if field_options.primary_key {
            if explicit_primary_key.is_some() {
                return Err(Error::new_spanned(
                    field_name,
                    "ModelSelect supports exactly one primary key field",
                ));
            }
            explicit_primary_key = Some(primary_key);
        } else if field_name_string(&field_name) == "id" {
            inferred_primary_key = Some(primary_key);
        }

        variants.push(variant.clone());
        model_columns.push(column_literal.clone());
        model_initializers.push(quote! {
            #field_name: row.get(offset + #field_index)?
        });
        let query_error = LitStr::new(
            &format!("should read {model}.{field_name} from database row"),
            Span::call_site(),
        );
        query_initializers.push(quote! {
            #field_name: row.get(#column_literal).expect(#query_error)
        });
        column_constants.push(quote! {
            #[expect(
                non_upper_case_globals,
                reason = "selector field constants mirror generated Rust field variants"
            )]
            pub const #variant: ::gen_models::select::SelectField<#model, #field_type> =
                ::gen_models::select::SelectField::new(
                    <#model as ::gen_models::traits::Query>::TABLE_NAME,
                    #source_alias,
                    #column_literal,
                );
        });
        filter_methods.push(filter_method(
            &field_name,
            value_type,
            &column_literal,
            is_string(value_type),
            option_inner(&field.ty).is_some(),
        ));
    }

    if variants.is_empty() {
        return Err(Error::new_spanned(
            model,
            "ModelSelect requires at least one selectable field",
        ));
    }

    let primary_key = explicit_primary_key.or(inferred_primary_key);
    let primary_key_methods = primary_key
        .as_ref()
        .map(|(field, field_type, string, _)| {
            let in_method = format_ident!("{}_in", field);
            if *string {
                quote! {
                    pub fn get_by_id(
                        self,
                        id: impl ::core::convert::Into<::std::string::String>,
                    ) -> ::core::result::Result<
                        ::core::option::Option<#model>,
                        ::gen_models::ModelSelectError,
                    > {
                        self.#field(id).get()
                    }

                    pub fn query_by_ids<I, V>(
                        self,
                        ids: I,
                    ) -> ::core::result::Result<
                        ::std::vec::Vec<#model>,
                        ::gen_models::ModelSelectError,
                    >
                    where
                        I: ::core::iter::IntoIterator<Item = V>,
                        V: ::core::convert::Into<::std::string::String>,
                    {
                        self.#in_method(ids).load()
                    }
                }
            } else {
                quote! {
                    pub fn get_by_id(
                        self,
                        id: #field_type,
                    ) -> ::core::result::Result<
                        ::core::option::Option<#model>,
                        ::gen_models::ModelSelectError,
                    > {
                        self.#field(id).get()
                    }

                    pub fn query_by_ids<I>(
                        self,
                        ids: I,
                    ) -> ::core::result::Result<
                        ::std::vec::Vec<#model>,
                        ::gen_models::ModelSelectError,
                    >
                    where
                        I: ::core::iter::IntoIterator<Item = #field_type>,
                    {
                        self.#in_method(ids).load()
                    }
                }
            }
        })
        .unwrap_or_default();

    let query_impl = if let Some(table) = options.table.as_ref() {
        let primary_key = primary_key
            .as_ref()
            .map(|(_, _, _, column)| {
                quote! {
                    const PRIMARY_KEY: &'static str = #column;
                }
            })
            .unwrap_or_default();
        let history_table_name = if options.history.unwrap_or(true) {
            quote! {
                const HISTORY_TABLE_NAME: ::core::option::Option<&'static str> =
                    ::core::option::Option::Some(Self::TABLE_NAME);
            }
        } else {
            quote! {
                const HISTORY_TABLE_NAME: ::core::option::Option<&'static str> =
                    ::core::option::Option::None;
            }
        };
        let process_row = if let Some(from_row) = options.from_row.as_ref() {
            quote! { #from_row(row) }
        } else {
            if has_skipped_field {
                return Err(Error::new_spanned(
                    model,
                    "ModelSelect requires `from_row = path` when a table-backed model skips fields",
                ));
            }
            quote! {
                Self {
                    #(#query_initializers),*
                }
            }
        };
        quote! {
            impl ::gen_models::traits::Query for #model {
                type Model = Self;

                #primary_key
                const TABLE_NAME: &'static str = #table;
                #history_table_name

                fn process_row(
                    row: &::gen_models::select::Row,
                ) -> Self::Model {
                    #process_row
                }
            }
        }
    } else {
        if options.from_row.is_some() || options.history.is_some() {
            return Err(Error::new_spanned(
                model,
                "ModelSelect `from_row` and `history` options require `table = \"...\"`",
            ));
        }
        quote! {}
    };

    let source_clause = if let Some(source) = options.source.as_ref() {
        quote! { #source(history_ref) }
    } else {
        quote! {
            ::gen_models::select::default_sql_source(
                <#model as ::gen_models::traits::Query>::TABLE_NAME,
                <#model as ::gen_models::traits::Query>::HISTORY_TABLE_NAME,
                #source_alias,
                history_ref,
            )
        }
    };
    let select_clause = options
        .select
        .as_ref()
        .map(|select| quote! { ::std::string::String::from(#select) })
        .unwrap_or_else(|| {
            quote! {
                ::gen_models::select::select_all_sql(
                    <Self as ::gen_models::select::SelectQuery>::source(self).alias(),
                )
            }
        });
    let selectable_model_impl = if has_skipped_field {
        quote! {}
    } else {
        quote! {
            impl ::gen_models::select::SelectableModel for #model {
                fn table_name() -> &'static str {
                    <Self as ::gen_models::traits::Query>::TABLE_NAME
                }

                fn alias() -> &'static str {
                    #source_alias
                }

                fn columns() -> &'static [&'static str] {
                    &[#(#model_columns),*]
                }

                fn process_row(
                    row: &::gen_models::select::Row,
                    offset: usize,
                ) -> ::gen_models::select::SqlResult<Self> {
                    ::core::result::Result::Ok(Self {
                        #(#model_initializers),*
                    })
                }
            }
        }
    };

    Ok(quote! {
        #query_impl
        #selectable_model_impl

        #[derive(Clone, Debug)]
        pub struct #selector<'conn> {
            conn: &'conn ::gen_models::select::Connection,
            history_ref: ::core::option::Option<::std::string::String>,
            filters: ::std::vec::Vec<::gen_models::select::SqlFilter>,
            order_by: ::std::vec::Vec<::gen_models::select::SqlOrder>,
            joins: ::gen_models::select::SqlJoins,
            limit: ::core::option::Option<u32>,
            offset: u32,
        }

        impl<'conn> #selector<'conn> {
            #(#column_constants)*

            pub fn new(
                conn: &'conn ::gen_models::select::Connection,
            ) -> Self {
                Self {
                    conn,
                    history_ref: ::core::option::Option::None,
                    filters: ::std::vec::Vec::new(),
                    order_by: ::std::vec::Vec::new(),
                    joins: ::gen_models::select::SqlJoins::default(),
                    limit: ::core::option::Option::None,
                    offset: 0,
                }
            }

            fn source_clause_for(history_ref: ::core::option::Option<&str>) -> ::std::string::String {
                #source_clause
            }

            fn column(&self, column: &str) -> ::std::string::String {
                ::gen_models::select::qualify_sql_column(
                    <Self as ::gen_models::select::SelectQuery>::source(self).alias(),
                    column,
                )
            }

            pub fn with_ref(mut self, history_ref: impl ::core::convert::Into<::std::string::String>) -> Self {
                self.history_ref = ::core::option::Option::Some(history_ref.into());
                self
            }

            #(#filter_methods)*

            pub fn order_by<T>(
                mut self,
                field: ::gen_models::select::SelectField<#model, T>,
                direction: ::gen_models::select::Direction,
            ) -> Self {
                let column = self.column(field.column());
                self.order_by.push(::gen_models::select::SqlOrder::new(
                    column,
                    direction,
                ));
                self
            }

            pub fn join_on<SourceType, JoinedModel, JoinedType>(
                self,
                source_field: ::gen_models::select::SelectField<#model, SourceType>,
                joined_field: ::gen_models::select::SelectField<JoinedModel, JoinedType>,
            ) -> Self
            where
                JoinedModel: ::gen_models::select::ModelSelectSource,
            {
                let joined =
                    <JoinedModel as ::gen_models::select::ModelSelectSource>::selector(self.conn);
                self.join_filtered_on(source_field, joined_field, joined)
            }

            pub fn join_filtered_on<SourceType, JoinedModel, JoinedType>(
                mut self,
                source_field: ::gen_models::select::SelectField<#model, SourceType>,
                joined_field: ::gen_models::select::SelectField<JoinedModel, JoinedType>,
                joined: <JoinedModel as ::gen_models::select::ModelSelectSource>::Selector<'conn>,
            ) -> Self
            where
                JoinedModel: ::gen_models::select::ModelSelectSource,
            {
                assert!(
                    ::core::ptr::eq(
                        self.conn,
                        ::gen_models::select::SelectQuery::connection(&joined),
                    ),
                    "joined selectors must use the same database connection",
                );
                assert!(
                    ::gen_models::select::SelectQuery::limit(&joined).is_none()
                        && ::gen_models::select::SelectQuery::offset(&joined) == 0,
                    "apply limit and offset after join, not to the joined selector",
                );

                if let ::core::option::Option::Some(joined_ref) =
                    ::gen_models::select::SelectQuery::history_ref(&joined)
                {
                    if let ::core::option::Option::Some(history_ref) = &self.history_ref {
                        assert!(
                            history_ref == joined_ref,
                            "joined selectors must use the same historical ref",
                        );
                    } else {
                        self.history_ref = ::core::option::Option::Some(joined_ref.to_string());
                    }
                }

                let mut existing_sources = ::std::vec::Vec::with_capacity(self.joins.len() + 1);
                existing_sources.push(
                    <Self as ::gen_models::select::SelectQuery>::source(&self),
                );
                existing_sources.extend(self.joins.iter().map(|join| join.source()));

                let joined_source = ::gen_models::select::SelectQuery::source(&joined);
                let explicit_join = ::gen_models::select::sql_join_on(
                    &existing_sources,
                    joined_source,
                    source_field,
                    joined_field,
                );

                let base_alias =
                    <Self as ::gen_models::select::SelectQuery>::source(&self).alias();
                for nested_join in
                    ::gen_models::select::SelectQuery::joins(&joined).iter()
                {
                    assert!(
                        nested_join.source().alias() != base_alias,
                        "cannot join the base SQL source alias `{base_alias}`",
                    );
                }
                self.joins.insert(explicit_join);
                self.joins.extend_from(
                    ::gen_models::select::SelectQuery::joins(&joined),
                );
                self.filters.extend_from_slice(
                    ::gen_models::select::SelectQuery::filters(&joined),
                );
                self.order_by.extend_from_slice(
                    ::gen_models::select::SelectQuery::order_by(&joined),
                );
                self
            }

            pub fn only<P>(
                self,
                projection: P,
            ) -> ::gen_models::select::SelectedFields<Self, P>
            where
                P: ::gen_models::select::SelectProjection,
            {
                ::gen_models::select::SelectedFields::new(self, projection)
            }

            pub fn models<P>(
                self,
            ) -> ::gen_models::select::SelectedModels<Self, P>
            where
                P: ::gen_models::select::ModelProjection,
            {
                ::gen_models::select::SelectedModels::new(self)
            }

            pub fn limit(mut self, limit: u32) -> Self {
                self.limit = ::core::option::Option::Some(limit);
                self
            }

            pub fn offset(mut self, offset: u32) -> Self {
                self.offset = offset;
                self
            }

            pub fn load(
                self,
            ) -> ::core::result::Result<
                ::std::vec::Vec<#model>,
                ::gen_models::ModelSelectError,
            > {
                ::gen_models::select::load::<#model, _>(self.conn, &self)
            }

            pub fn get(
                mut self,
            ) -> ::core::result::Result<
                ::core::option::Option<#model>,
                ::gen_models::ModelSelectError,
            > {
                self.limit = ::core::option::Option::Some(2);
                ::gen_models::select::get::<#model, _>(self.conn, &self)
            }

            #primary_key_methods

            pub(crate) fn push_filter(
                mut self,
                filter: ::gen_models::select::SqlFilter,
            ) -> Self {
                self.filters.push(filter);
                self
            }
        }

        impl ::gen_models::select::SelectQuery for #selector<'_> {
            fn connection(&self) -> &::gen_models::select::Connection {
                self.conn
            }

            fn source(&self) -> ::gen_models::select::SqlSource {
                ::gen_models::select::SqlSource::new(
                    <#model as ::gen_models::traits::Query>::TABLE_NAME,
                    #source_alias,
                    ::core::concat!(::core::module_path!(), "::", ::core::stringify!(#model)),
                    Self::source_clause_for,
                )
            }

            fn history_ref(&self) -> ::core::option::Option<&str> {
                self.history_ref.as_deref()
            }

            fn joins(&self) -> &::gen_models::select::SqlJoins {
                &self.joins
            }

            fn select_clause(&self) -> ::std::string::String {
                #select_clause
            }

            fn filters(&self) -> &[::gen_models::select::SqlFilter] {
                &self.filters
            }

            fn order_by(&self) -> &[::gen_models::select::SqlOrder] {
                &self.order_by
            }

            fn limit(&self) -> ::core::option::Option<u32> {
                self.limit
            }

            fn offset(&self) -> u32 {
                self.offset
            }
        }

        impl ::gen_models::select::ModelSelectSource for #model {
            type Selector<'conn> = #selector<'conn>;

            fn selector(
                conn: &::gen_models::select::Connection,
            ) -> Self::Selector<'_> {
                #selector::new(conn)
            }
        }

        impl #model {
            pub fn select(
                conn: &::gen_models::select::Connection,
            ) -> #selector<'_> {
                #selector::new(conn)
            }

            pub fn all(
                conn: &::gen_models::select::Connection,
            ) -> ::core::result::Result<
                ::std::vec::Vec<Self>,
                ::gen_models::ModelSelectError,
            > {
                Self::select(conn).load()
            }
        }
    })
}

#[derive(Default)]
struct ContainerOptions {
    alias: Option<String>,
    table: Option<LitStr>,
    history: Option<bool>,
    from_row: Option<ExprPath>,
    source: Option<ExprPath>,
    select: Option<LitStr>,
}

impl ContainerOptions {
    fn from_attributes(attributes: &[Attribute]) -> syn::Result<Self> {
        let mut options = Self::default();
        for attribute in attributes
            .iter()
            .filter(|attribute| attribute.path().is_ident("model_select"))
        {
            attribute.parse_nested_meta(|meta| {
                if meta.path.is_ident("alias") {
                    let alias: LitStr = meta.value()?.parse()?;
                    options.alias = Some(alias.value());
                    return Ok(());
                }
                if meta.path.is_ident("table") {
                    options.table = Some(meta.value()?.parse()?);
                    return Ok(());
                }
                if meta.path.is_ident("history") {
                    let history: LitBool = meta.value()?.parse()?;
                    options.history = Some(history.value());
                    return Ok(());
                }
                if meta.path.is_ident("from_row") {
                    options.from_row = Some(meta.value()?.parse()?);
                    return Ok(());
                }
                if meta.path.is_ident("source") {
                    options.source = Some(meta.value()?.parse()?);
                    return Ok(());
                }
                if meta.path.is_ident("select") {
                    options.select = Some(meta.value()?.parse()?);
                    return Ok(());
                }
                Err(meta.error("unsupported model_select option"))
            })?;
        }
        Ok(options)
    }
}

#[derive(Default)]
struct FieldOptions {
    column: Option<String>,
    primary_key: bool,
    skip: bool,
}

impl FieldOptions {
    fn from_attributes(attributes: &[Attribute]) -> syn::Result<Self> {
        let mut options = Self::default();
        for attribute in attributes
            .iter()
            .filter(|attribute| attribute.path().is_ident("model_select"))
        {
            attribute.parse_nested_meta(|meta| {
                if meta.path.is_ident("column") {
                    let column: LitStr = meta.value()?.parse()?;
                    options.column = Some(column.value());
                    return Ok(());
                }
                if meta.path.is_ident("skip") {
                    options.skip = true;
                    return Ok(());
                }
                if meta.path.is_ident("primary_key") {
                    options.primary_key = true;
                    return Ok(());
                }
                Err(meta.error("unsupported model_select field option"))
            })?;
        }
        Ok(options)
    }
}

fn filter_method(
    field: &Ident,
    value_type: &Type,
    column: &LitStr,
    string: bool,
    optional: bool,
) -> proc_macro2::TokenStream {
    let in_method = format_ident!("{}_in", field);
    let exact = if string {
        quote! {
            pub fn #field(mut self, value: impl ::core::convert::Into<::std::string::String>) -> Self {
                let value = value.into();
                let column = self.column(#column);
                self.filters.push(::gen_models::select::SqlFilter::new(
                    ::std::format!("{column} = ?"),
                    ::std::vec![::gen_models::select::sql_value(&value)],
                ));
                self
            }
        }
    } else {
        quote! {
            pub fn #field(mut self, value: #value_type) -> Self {
                let column = self.column(#column);
                self.filters.push(::gen_models::select::SqlFilter::new(
                    ::std::format!("{column} = ?"),
                    ::std::vec![::gen_models::select::sql_value(&value)],
                ));
                self
            }
        }
    };

    let any_of = if string {
        quote! {
            pub fn #in_method<I, V>(mut self, values: I) -> Self
            where
                I: ::core::iter::IntoIterator<Item = V>,
                V: ::core::convert::Into<::std::string::String>,
            {
                let column = self.column(#column);
                let params = values
                    .into_iter()
                    .map(|value| {
                        let value = value.into();
                        ::gen_models::select::sql_value(&value)
                    })
                    .collect::<::std::vec::Vec<_>>();
                self.filters.push(::gen_models::select::sql_in_filter(column, params));
                self
            }
        }
    } else {
        quote! {
            pub fn #in_method<I>(mut self, values: I) -> Self
            where
                I: ::core::iter::IntoIterator<Item = #value_type>,
            {
                let column = self.column(#column);
                let params = values
                    .into_iter()
                    .map(|value| ::gen_models::select::sql_value(&value))
                    .collect::<::std::vec::Vec<_>>();
                self.filters.push(::gen_models::select::sql_in_filter(column, params));
                self
            }
        }
    };

    let contains = if string {
        let contains_method = format_ident!("{}_contains", field);
        quote! {
            pub fn #contains_method(
                mut self,
                value: impl ::core::convert::Into<::std::string::String>,
            ) -> Self {
                let value = value.into();
                let column = self.column(#column);
                self.filters.push(::gen_models::select::SqlFilter::new(
                    ::std::format!("instr(lower({column}), lower(?)) > 0"),
                    ::std::vec![::gen_models::select::sql_value(&value)],
                ));
                self
            }
        }
    } else {
        quote! {}
    };

    let case_insensitive = if string {
        let case_insensitive_method = format_ident!("{}_case_insensitive", field);
        quote! {
            pub fn #case_insensitive_method(
                mut self,
                value: impl ::core::convert::Into<::std::string::String>,
            ) -> Self {
                let value = value.into();
                let column = self.column(#column);
                self.filters.push(::gen_models::select::SqlFilter::new(
                    ::std::format!("lower({column}) = lower(?)"),
                    ::std::vec![::gen_models::select::sql_value(&value)],
                ));
                self
            }
        }
    } else {
        quote! {}
    };

    let is_null = if optional {
        let is_null_method = format_ident!("{}_is_null", field);
        quote! {
            pub fn #is_null_method(mut self) -> Self {
                let column = self.column(#column);
                self.filters.push(::gen_models::select::SqlFilter::new(
                    ::std::format!("{column} IS NULL"),
                    ::std::vec::Vec::new(),
                ));
                self
            }
        }
    } else {
        quote! {}
    };

    quote! {
        #exact
        #any_of
        #contains
        #case_insensitive
        #is_null
    }
}

fn option_inner(field_type: &Type) -> Option<&Type> {
    let Type::Path(path) = field_type else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    let syn::GenericArgument::Type(inner) = arguments.args.first()? else {
        return None;
    };
    Some(inner)
}

fn is_string(field_type: &Type) -> bool {
    let Type::Path(path) = field_type else {
        return false;
    };
    path.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "String")
}

fn snake_to_pascal(identifier: &Ident) -> Ident {
    let name = field_name_string(identifier);
    let pascal = name
        .split('_')
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut characters = part.chars();
            characters
                .next()
                .map(|first| first.to_uppercase().collect::<String>() + characters.as_str())
                .unwrap_or_default()
        })
        .collect::<String>();
    Ident::new(&pascal, identifier.span())
}

fn field_name_string(identifier: &Ident) -> String {
    let name = identifier.to_string();
    name.strip_prefix("r#").unwrap_or(&name).to_string()
}
