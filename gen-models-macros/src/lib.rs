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
/// Primary-key selectors also provide `get_by_id(...)`, ordered and deduplicated
/// `query_by_ids(...)` loads, and batched `delete_by_ids(...)` mutations.
/// `#[model_select(table = "...")]` generates the model's `Query` implementation.
/// `#[model_select(column = "...")]` overrides a field's SQL column,
/// `#[model_select(primary_key)]` marks each field in an explicit primary key,
/// `#[model_select(default_sort = "asc")]` configures default ordering,
/// `#[model_select(skip)]` excludes a field, and the struct-level `history`, `from_row`, `source`,
/// `alias`, and `select` options support custom persistence behavior or aliased queries.
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
    let mut explicit_primary_keys = Vec::new();
    let mut inferred_primary_key = None;
    let mut default_order_initializers = Vec::new();

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
            if field_options.default_sort.is_some() {
                return Err(Error::new_spanned(
                    field_name,
                    "a skipped ModelSelect field cannot define a default sort",
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

        let primary_key = PrimaryKeyField {
            field: field_name.clone(),
            value_type: value_type.clone(),
            string: is_string(value_type),
            column: column_literal.clone(),
        };
        if field_options.primary_key {
            explicit_primary_keys.push(primary_key);
        } else if field_name_string(&field_name) == "id" {
            inferred_primary_key = Some(primary_key);
        }
        if let Some(direction) = field_options.default_sort {
            let direction = direction.tokens();
            default_order_initializers.push(quote! {
                ::gen_models::select::SqlOrder::new(
                    ::gen_models::select::qualify_sql_column(#source_alias, #column_literal),
                    #direction,
                )
            });
        }

        variants.push(variant.clone());
        model_columns.push(column_literal.clone());
        model_initializers.push(quote! {
            #field_name: row.get(offset + #field_index)?
        });
        query_initializers.push(quote! {
            #field_name: row.get(#column_literal)?
        });
        column_constants.push(quote! {
            #[expect(
                non_upper_case_globals,
                reason = "selector field constants mirror generated Rust field variants"
            )]
            pub const #variant: ::gen_models::select::SelectField<#model, #field_type, #value_type> =
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

    let primary_keys = if explicit_primary_keys.is_empty() {
        inferred_primary_key.into_iter().collect::<Vec<_>>()
    } else {
        explicit_primary_keys
    };
    let primary_key_methods = primary_key_methods(&model, &primary_keys);

    let query_impl = if let Some(table) = options.table.as_ref() {
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
                ::core::result::Result::Ok(Self {
                    #(#query_initializers),*
                })
            }
        };
        quote! {
            impl ::gen_models::traits::Query for #model {
                type Model = Self;

                const TABLE_NAME: &'static str = #table;
                #history_table_name

                fn process_row(
                    row: &::gen_models::select::Row,
                ) -> ::gen_models::select::SqlResult<Self::Model> {
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
            error: ::core::option::Option<::gen_models::select::SelectorBuildError>,
            history_ref: ::core::option::Option<::std::string::String>,
            filters: ::std::vec::Vec<::gen_models::select::SqlFilter>,
            default_order_by: ::std::vec::Vec<::gen_models::select::SqlOrder>,
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
                    error: ::core::option::Option::None,
                    history_ref: ::core::option::Option::None,
                    filters: ::std::vec::Vec::new(),
                    default_order_by: ::std::vec![#(#default_order_initializers),*],
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

            fn invalid_selector(mut self, message: impl ::core::convert::Into<::std::string::String>) -> Self {
                if self.error.is_none() {
                    self.error = ::core::option::Option::Some(
                        ::gen_models::select::SelectorBuildError::InvalidSelector(message.into()),
                    );
                }
                self
            }

            fn validate_delete_by_ids(
                &self,
            ) -> ::core::result::Result<(), ::gen_models::ModelSelectError> {
                if let ::core::option::Option::Some(error) = &self.error {
                    return ::core::result::Result::Err(error.clone().into());
                }
                if self.history_ref.is_some()
                    || !self.filters.is_empty()
                    || !self.order_by.is_empty()
                    || !self.joins.is_empty()
                    || self.limit.is_some()
                    || self.offset != 0
                {
                    return ::core::result::Result::Err(
                        ::gen_models::ModelSelectError::InvalidSelector(
                            ::std::string::String::from(
                                "delete_by_ids must be called before configuring the selector",
                            ),
                        ),
                    );
                }
                ::core::result::Result::Ok(())
            }

            fn push_filter_result(
                mut self,
                filter: ::core::result::Result<
                    ::gen_models::select::SqlFilter,
                    ::gen_models::select::SelectorBuildError,
                >,
            ) -> Self {
                match filter {
                    ::core::result::Result::Ok(filter) => self.filters.push(filter),
                    ::core::result::Result::Err(error) if self.error.is_none() => {
                        self.error = ::core::option::Option::Some(error);
                    }
                    ::core::result::Result::Err(_) => {}
                }
                self
            }

            pub fn with_ref<R, K>(
                mut self,
                history_ref: R,
            ) -> Self
            where
                R: ::gen_models::select::IntoHistoryRef<K>,
            {
                self.history_ref =
                    ::gen_models::select::IntoHistoryRef::into_history_ref(history_ref);
                self
            }

            #(#filter_methods)*

            pub fn order_by<T, JoinType>(
                mut self,
                field: ::gen_models::select::SelectField<#model, T, JoinType>,
                direction: ::gen_models::select::Direction,
            ) -> Self {
                let column = self.column(field.column());
                self.order_by.push(::gen_models::select::SqlOrder::new(
                    column,
                    direction,
                ));
                self
            }

            pub fn join_on<SourceType, JoinedModel, JoinedType, JoinType>(
                self,
                source_field: ::gen_models::select::SelectField<#model, SourceType, JoinType>,
                joined_field: ::gen_models::select::SelectField<JoinedModel, JoinedType, JoinType>,
            ) -> Self
            where
                JoinedModel: ::gen_models::select::ModelSelectSource,
            {
                let joined =
                    <JoinedModel as ::gen_models::select::ModelSelectSource>::selector(self.conn);
                self.join_filtered_on(source_field, joined_field, joined)
            }

            pub fn join_filtered_on<SourceType, JoinedModel, JoinedType, JoinType>(
                mut self,
                source_field: ::gen_models::select::SelectField<#model, SourceType, JoinType>,
                joined_field: ::gen_models::select::SelectField<JoinedModel, JoinedType, JoinType>,
                joined: <JoinedModel as ::gen_models::select::ModelSelectSource>::Selector<'conn>,
            ) -> Self
            where
                JoinedModel: ::gen_models::select::ModelSelectSource,
            {
                if self.error.is_some() {
                    return self;
                }
                if let ::core::option::Option::Some(error) =
                    ::gen_models::select::SelectQuery::error(&joined)
                {
                    self.error = ::core::option::Option::Some(error.clone());
                    return self;
                }
                if !::core::ptr::eq(
                    self.conn,
                    ::gen_models::select::SelectQuery::connection(&joined),
                ) {
                    return self.invalid_selector(
                        "joined selectors must use the same database connection",
                    );
                }
                if ::gen_models::select::SelectQuery::limit(&joined).is_some()
                    || ::gen_models::select::SelectQuery::offset(&joined) != 0
                {
                    return self.invalid_selector(
                        "apply limit and offset after join, not to the joined selector",
                    );
                }

                if let ::core::option::Option::Some(joined_ref) =
                    ::gen_models::select::SelectQuery::history_ref(&joined)
                {
                    if let ::core::option::Option::Some(history_ref) = &self.history_ref {
                        if history_ref != joined_ref {
                            return self.invalid_selector(
                                "joined selectors must use the same historical ref",
                            );
                        }
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
                let explicit_join = match ::gen_models::select::sql_join_on(
                    &existing_sources,
                    joined_source,
                    source_field,
                    joined_field,
                ) {
                    ::core::result::Result::Ok(join) => join,
                    ::core::result::Result::Err(error) => {
                        self.error = ::core::option::Option::Some(error);
                        return self;
                    }
                };

                let base_alias =
                    <Self as ::gen_models::select::SelectQuery>::source(&self).alias();
                for nested_join in
                    ::gen_models::select::SelectQuery::joins(&joined).iter()
                {
                    if nested_join.source().alias() == base_alias {
                        return self.invalid_selector(::std::format!(
                            "cannot join the base SQL source alias `{base_alias}`",
                        ));
                    }
                }
                if let ::core::result::Result::Err(error) = self.joins.insert(explicit_join) {
                    self.error = ::core::option::Option::Some(error);
                    return self;
                }
                if let ::core::result::Result::Err(error) = self.joins.extend_from(
                    ::gen_models::select::SelectQuery::joins(&joined),
                ) {
                    self.error = ::core::option::Option::Some(error);
                    return self;
                }
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
                self,
            ) -> ::core::result::Result<
                ::core::option::Option<#model>,
                ::gen_models::ModelSelectError,
            > {
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
                    <#model as ::gen_models::traits::Query>::HISTORY_TABLE_NAME.is_some(),
                    Self::source_clause_for,
                )
            }

            fn error(
                &self,
            ) -> ::core::option::Option<&::gen_models::select::SelectorBuildError> {
                self.error.as_ref()
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

            fn default_order_by(&self) -> &[::gen_models::select::SqlOrder] {
                &self.default_order_by
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

#[derive(Clone)]
struct PrimaryKeyField {
    field: Ident,
    value_type: Type,
    string: bool,
    column: LitStr,
}

#[derive(Clone, Copy)]
enum DefaultSort {
    Asc,
    Desc,
    CaseInsensitiveAsc,
    CaseInsensitiveDesc,
}

impl DefaultSort {
    fn from_literal(literal: &LitStr) -> syn::Result<Self> {
        match literal.value().as_str() {
            "asc" => Ok(Self::Asc),
            "desc" => Ok(Self::Desc),
            "case_insensitive_asc" => Ok(Self::CaseInsensitiveAsc),
            "case_insensitive_desc" => Ok(Self::CaseInsensitiveDesc),
            _ => Err(Error::new_spanned(
                literal,
                "default_sort must be `asc`, `desc`, `case_insensitive_asc`, or `case_insensitive_desc`",
            )),
        }
    }

    fn tokens(self) -> proc_macro2::TokenStream {
        match self {
            Self::Asc => quote! { ::gen_models::Direction::Asc },
            Self::Desc => quote! { ::gen_models::Direction::Desc },
            Self::CaseInsensitiveAsc => {
                quote! { ::gen_models::Direction::CaseInsensitiveAsc }
            }
            Self::CaseInsensitiveDesc => {
                quote! { ::gen_models::Direction::CaseInsensitiveDesc }
            }
        }
    }
}

#[derive(Default)]
struct FieldOptions {
    column: Option<String>,
    default_sort: Option<DefaultSort>,
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
                if meta.path.is_ident("default_sort") {
                    if options.default_sort.is_some() {
                        return Err(meta.error("default_sort can only be specified once per field"));
                    }
                    options.default_sort = if meta.input.peek(syn::Token![=]) {
                        let literal: LitStr = meta.value()?.parse()?;
                        Some(DefaultSort::from_literal(&literal)?)
                    } else {
                        Some(DefaultSort::Asc)
                    };
                    return Ok(());
                }
                Err(meta.error("unsupported model_select field option"))
            })?;
        }
        Ok(options)
    }
}

fn primary_key_methods(
    model: &Ident,
    primary_keys: &[PrimaryKeyField],
) -> proc_macro2::TokenStream {
    match primary_keys {
        [] => quote! {},
        [primary_key] => single_primary_key_methods(model, primary_key),
        primary_keys => composite_primary_key_methods(model, primary_keys),
    }
}

fn single_primary_key_methods(
    model: &Ident,
    primary_key: &PrimaryKeyField,
) -> proc_macro2::TokenStream {
    let PrimaryKeyField {
        field,
        value_type,
        string,
        column,
    } = primary_key;
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

            pub fn delete_by_ids<I, V>(
                self,
                ids: I,
            ) -> ::core::result::Result<usize, ::gen_models::ModelSelectError>
            where
                I: ::core::iter::IntoIterator<Item = V>,
                V: ::core::convert::Into<::std::string::String>,
            {
                self.validate_delete_by_ids()?;
                let rows = ids
                    .into_iter()
                    .map(|id| {
                        let id = id.into();
                        ::gen_models::select::sql_value(&id).map(|value| ::std::vec![value])
                    })
                    .collect::<::core::result::Result<::std::vec::Vec<_>, _>>()?;
                ::gen_models::select::delete_by_ids(
                    self.conn,
                    <#model as ::gen_models::traits::Query>::TABLE_NAME,
                    &[#column],
                    rows,
                )
            }
        }
    } else {
        quote! {
            pub fn get_by_id(
                self,
                id: #value_type,
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
                I: ::core::iter::IntoIterator<Item = #value_type>,
            {
                self.#in_method(ids).load()
            }

            pub fn delete_by_ids<I>(
                self,
                ids: I,
            ) -> ::core::result::Result<usize, ::gen_models::ModelSelectError>
            where
                I: ::core::iter::IntoIterator<Item = #value_type>,
            {
                self.validate_delete_by_ids()?;
                let rows = ids
                    .into_iter()
                    .map(|id| {
                        ::gen_models::select::sql_value(&id).map(|value| ::std::vec![value])
                    })
                    .collect::<::core::result::Result<::std::vec::Vec<_>, _>>()?;
                ::gen_models::select::delete_by_ids(
                    self.conn,
                    <#model as ::gen_models::traits::Query>::TABLE_NAME,
                    &[#column],
                    rows,
                )
            }
        }
    }
}

fn composite_primary_key_methods(
    model: &Ident,
    primary_keys: &[PrimaryKeyField],
) -> proc_macro2::TokenStream {
    let parameter_names = (0..primary_keys.len())
        .map(|index| format_ident!("primary_key_{index}"))
        .collect::<Vec<_>>();
    let string_generics = primary_keys
        .iter()
        .enumerate()
        .filter(|(_, primary_key)| primary_key.string)
        .map(|(index, _)| format_ident!("PrimaryKeyValue{index}"))
        .collect::<Vec<_>>();
    let mut string_generic_index = 0;
    let input_types = primary_keys
        .iter()
        .map(|primary_key| {
            if primary_key.string {
                let generic = &string_generics[string_generic_index];
                string_generic_index += 1;
                quote! { #generic }
            } else {
                let value_type = &primary_key.value_type;
                quote! { #value_type }
            }
        })
        .collect::<Vec<_>>();
    let string_bounds = string_generics
        .iter()
        .map(|generic| {
            quote! { #generic: ::core::convert::Into<::std::string::String> }
        })
        .collect::<Vec<_>>();
    let get_generics = if string_generics.is_empty() {
        quote! {}
    } else {
        quote! { <#(#string_generics),*> }
    };
    let get_where = if string_bounds.is_empty() {
        quote! {}
    } else {
        quote! { where #(#string_bounds),* }
    };
    let fields = primary_keys
        .iter()
        .map(|primary_key| &primary_key.field)
        .collect::<Vec<_>>();
    let columns = primary_keys
        .iter()
        .map(|primary_key| &primary_key.column)
        .collect::<Vec<_>>();
    let filter_chain =
        fields
            .iter()
            .zip(&parameter_names)
            .fold(quote! { self }, |chain, (field, parameter)| {
                quote! { #chain.#field(#parameter) }
            });
    let sql_values = primary_keys
        .iter()
        .zip(&parameter_names)
        .map(|(primary_key, parameter)| {
            if primary_key.string {
                quote! {{
                    let value = ::core::convert::Into::<::std::string::String>::into(#parameter);
                    ::gen_models::select::sql_value(&value)?
                }}
            } else {
                quote! { ::gen_models::select::sql_value(&#parameter)? }
            }
        })
        .collect::<Vec<_>>();

    quote! {
        pub fn get_by_id #get_generics(
            self,
            id: (#(#input_types,)*),
        ) -> ::core::result::Result<
            ::core::option::Option<#model>,
            ::gen_models::ModelSelectError,
        >
        #get_where
        {
            let (#(#parameter_names,)*) = id;
            #filter_chain.get()
        }

        pub fn query_by_ids<I #(, #string_generics)*>(
            self,
            ids: I,
        ) -> ::core::result::Result<
            ::std::vec::Vec<#model>,
            ::gen_models::ModelSelectError,
        >
        where
            I: ::core::iter::IntoIterator<Item = (#(#input_types,)*)>,
            #(#string_bounds,)*
        {
            let columns = ::std::vec![#(self.column(#columns)),*];
            let rows = ids
                .into_iter()
                .map(|(#(#parameter_names,)*)| {
                    ::core::result::Result::Ok(::std::vec![#(#sql_values),*])
                })
                .collect::<::core::result::Result<
                    ::std::vec::Vec<_>,
                    ::gen_models::select::SelectorBuildError,
                >>()?;
            let filter = ::gen_models::select::sql_composite_in_filter(columns, rows);
            self.push_filter_result(filter).load()
        }

        pub fn delete_by_ids<I #(, #string_generics)*>(
            self,
            ids: I,
        ) -> ::core::result::Result<usize, ::gen_models::ModelSelectError>
        where
            I: ::core::iter::IntoIterator<Item = (#(#input_types,)*)>,
            #(#string_bounds,)*
        {
            self.validate_delete_by_ids()?;
            let rows = ids
                .into_iter()
                .map(|(#(#parameter_names,)*)| {
                    ::core::result::Result::Ok(::std::vec![#(#sql_values),*])
                })
                .collect::<::core::result::Result<
                    ::std::vec::Vec<_>,
                    ::gen_models::select::SelectorBuildError,
                >>()?;
            ::gen_models::select::delete_by_ids(
                self.conn,
                <#model as ::gen_models::traits::Query>::TABLE_NAME,
                &[#(#columns),*],
                rows,
            )
        }
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
            pub fn #field(self, value: impl ::core::convert::Into<::std::string::String>) -> Self {
                let value = value.into();
                let column = self.column(#column);
                let filter = ::gen_models::select::sql_value(&value).map(|value| {
                    ::gen_models::select::SqlFilter::new(
                        ::std::format!("{column} = ?"),
                        ::std::vec![value],
                    )
                });
                self.push_filter_result(filter)
            }
        }
    } else {
        quote! {
            pub fn #field(self, value: #value_type) -> Self {
                let column = self.column(#column);
                let filter = ::gen_models::select::sql_value(&value).map(|value| {
                    ::gen_models::select::SqlFilter::new(
                        ::std::format!("{column} = ?"),
                        ::std::vec![value],
                    )
                });
                self.push_filter_result(filter)
            }
        }
    };

    let any_of = if string {
        quote! {
            pub fn #in_method<I, V>(self, values: I) -> Self
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
                    .collect::<::core::result::Result<::std::vec::Vec<_>, _>>();
                self.push_filter_result(
                    params.map(|params| ::gen_models::select::sql_in_filter(column, params)),
                )
            }
        }
    } else {
        quote! {
            pub fn #in_method<I>(self, values: I) -> Self
            where
                I: ::core::iter::IntoIterator<Item = #value_type>,
            {
                let column = self.column(#column);
                let params = values
                    .into_iter()
                    .map(|value| ::gen_models::select::sql_value(&value))
                    .collect::<::core::result::Result<::std::vec::Vec<_>, _>>();
                self.push_filter_result(
                    params.map(|params| ::gen_models::select::sql_in_filter(column, params)),
                )
            }
        }
    };

    let contains = if string {
        let contains_method = format_ident!("{}_contains", field);
        quote! {
            pub fn #contains_method(
                self,
                value: impl ::core::convert::Into<::std::string::String>,
            ) -> Self {
                let value = value.into();
                let column = self.column(#column);
                let filter = ::gen_models::select::sql_value(&value).map(|value| {
                    ::gen_models::select::SqlFilter::new(
                        ::std::format!("instr(lower({column}), lower(?)) > 0"),
                        ::std::vec![value],
                    )
                });
                self.push_filter_result(filter)
            }
        }
    } else {
        quote! {}
    };

    let case_insensitive = if string {
        let case_insensitive_method = format_ident!("{}_case_insensitive", field);
        quote! {
            pub fn #case_insensitive_method(
                self,
                value: impl ::core::convert::Into<::std::string::String>,
            ) -> Self {
                let value = value.into();
                let column = self.column(#column);
                let filter = ::gen_models::select::sql_value(&value).map(|value| {
                    ::gen_models::select::SqlFilter::new(
                        ::std::format!("lower({column}) = lower(?)"),
                        ::std::vec![value],
                    )
                });
                self.push_filter_result(filter)
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
