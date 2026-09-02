//! Derive support for typed, fluent SQL selectors in `gen-models`.

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::{
    Attribute, Data, DeriveInput, Error, ExprPath, Fields, Ident, LitStr, Type, parse_macro_input,
};

/// Generates a `<Model>Select` query builder from a persisted model's named fields.
///
/// String fields receive exact and `_contains` filters. Every generated field can be used for
/// typed ordering. Generated selectors can be composed with `.join(other_selector)` when their
/// models have one unambiguous direct foreign-key relationship. `#[model_select(column = "...")]`
/// overrides a field's SQL column, `#[model_select(skip)]` excludes a field, and the struct-level
/// `source`, `alias`, and `select` options support custom or aliased queries.
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
    let alias = options.alias.clone();
    let selector = format_ident!("{}Select", model);
    let selector_field = format_ident!("{}SelectField", model);

    let mut variants = Vec::new();
    let mut column_matches = Vec::new();
    let mut column_constants = Vec::new();
    let mut filter_methods = Vec::new();

    for field in fields {
        let Some(field_name) = field.ident else {
            continue;
        };
        let field_options = FieldOptions::from_attributes(&field.attrs)?;
        if field_options.skip {
            continue;
        }

        let variant = snake_to_pascal(&field_name);
        let column = field_options
            .column
            .unwrap_or_else(|| field_name_string(&field_name));
        let column_literal = LitStr::new(&column, Span::call_site());
        let value_type = option_inner(&field.ty).unwrap_or(&field.ty);

        variants.push(variant.clone());
        column_matches.push(quote! { Self::#variant => #column_literal });
        column_constants.push(quote! {
            #[expect(
                non_upper_case_globals,
                reason = "selector field constants mirror generated Rust field variants"
            )]
            pub const #variant: #selector_field = #selector_field::#variant;
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

    let source_clause = if let Some(source) = options.source {
        quote! { #source(history_ref) }
    } else {
        quote! {
            let source = <#model as ::gen_models::traits::Query>::table_name_with_history_ref(
                history_ref,
            );
            let alias = <#model as ::gen_models::traits::Query>::TABLE_NAME;
            ::std::format!("{source} {alias}")
        }
    };
    let source_alias = alias
        .map(|alias| {
            let alias = LitStr::new(&alias, Span::call_site());
            quote! { #alias }
        })
        .unwrap_or_else(|| {
            quote! { <#model as ::gen_models::traits::Query>::TABLE_NAME }
        });
    let select_clause = options
        .select
        .map(|select| quote! { ::std::string::String::from(#select) })
        .unwrap_or_else(|| {
            quote! {
                ::std::format!(
                    "{}.*",
                    <Self as ::gen_models::select::SelectQuery>::source(self).alias(),
                )
            }
        });

    Ok(quote! {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum #selector_field {
            #(#variants),*
        }

        impl #selector_field {
            const fn as_sql(self) -> &'static str {
                match self {
                    #(#column_matches),*
                }
            }
        }

        #[derive(Clone, Debug)]
        pub struct #selector<'conn> {
            conn: &'conn ::gen_models::select::Connection,
            history_ref: ::core::option::Option<::std::string::String>,
            filters: ::std::vec::Vec<::gen_models::select::SqlFilter>,
            order_by: ::std::vec::Vec<::gen_models::select::SqlOrder>,
            joins: ::std::vec::Vec<::gen_models::select::SqlJoin>,
            limit: ::core::option::Option<u32>,
            offset: u32,
        }

        impl<'conn> #selector<'conn> {
            #(#column_constants)*

            pub const fn new(
                conn: &'conn ::gen_models::select::Connection,
            ) -> Self {
                Self {
                    conn,
                    history_ref: ::core::option::Option::None,
                    filters: ::std::vec::Vec::new(),
                    order_by: ::std::vec::Vec::new(),
                    joins: ::std::vec::Vec::new(),
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

            pub fn order_by(
                mut self,
                field: #selector_field,
                direction: ::gen_models::select::Direction,
            ) -> Self {
                let column = self.column(field.as_sql());
                self.order_by.push(::gen_models::select::SqlOrder::new(
                    column,
                    direction,
                ));
                self
            }

            pub fn join<J: ::gen_models::select::SelectQuery>(mut self, joined: J) -> Self {
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
                let inferred_join = ::gen_models::select::infer_sql_join(
                    self.conn,
                    &existing_sources,
                    joined_source,
                );

                let mut used_aliases = existing_sources
                    .iter()
                    .map(|source| source.alias())
                    .chain(::core::iter::once(joined_source.alias()))
                    .collect::<::std::collections::HashSet<_>>();
                for nested_join in ::gen_models::select::SelectQuery::joins(&joined) {
                    assert!(
                        used_aliases.insert(nested_join.source().alias()),
                        "cannot join SQL source alias `{}` more than once",
                        nested_join.source().alias(),
                    );
                }

                self.joins.push(inferred_join);
                self.joins.extend_from_slice(
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

            pub fn limit(mut self, limit: u32) -> Self {
                self.limit = ::core::option::Option::Some(limit);
                self
            }

            pub fn offset(mut self, offset: u32) -> Self {
                self.offset = offset;
                self
            }

            pub fn load(self) -> ::std::vec::Vec<#model> {
                ::gen_models::select::load::<#model, _>(self.conn, &self)
            }

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
                    Self::source_clause_for,
                )
            }

            fn history_ref(&self) -> ::core::option::Option<&str> {
                self.history_ref.as_deref()
            }

            fn joins(&self) -> &[::gen_models::select::SqlJoin] {
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

        impl #model {
            pub fn select(
                conn: &::gen_models::select::Connection,
            ) -> #selector<'_> {
                #selector::new(conn)
            }
        }
    })
}

#[derive(Default)]
struct ContainerOptions {
    alias: Option<String>,
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
        #contains
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
