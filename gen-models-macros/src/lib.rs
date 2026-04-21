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
/// typed ordering. `#[model_select(column = "...")]` overrides a field's SQL column,
/// `#[model_select(skip)]` excludes a field, and the struct-level `source`, `alias`, and `select`
/// options support joined or aliased queries.
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
        let column = field_options.column.unwrap_or_else(|| {
            let field_name = field_name_string(&field_name);
            alias
                .as_ref()
                .map(|alias| format!("{alias}.{field_name}"))
                .unwrap_or(field_name)
        });
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
        quote! { #source(self.history_ref.as_deref()) }
    } else {
        quote! {
            <#model as ::gen_models::traits::Query>::table_name_with_history_ref(
                self.history_ref.as_deref(),
            )
        }
    };
    let select_clause = options.select.unwrap_or_else(|| {
        alias
            .map(|alias| LitStr::new(&format!("{alias}.*"), Span::call_site()))
            .unwrap_or_else(|| LitStr::new("*", Span::call_site()))
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
            conn: &'conn ::gen_models::traits::SelectConnection,
            history_ref: ::core::option::Option<::std::string::String>,
            filters: ::std::vec::Vec<::gen_models::traits::SqlFilter>,
            order_by: ::std::vec::Vec<::gen_models::traits::SqlOrder>,
            page: ::gen_models::traits::PageRequest,
        }

        impl<'conn> #selector<'conn> {
            #(#column_constants)*

            pub const fn new(
                conn: &'conn ::gen_models::traits::SelectConnection,
            ) -> Self {
                Self {
                    conn,
                    history_ref: ::core::option::Option::None,
                    filters: ::std::vec::Vec::new(),
                    order_by: ::std::vec::Vec::new(),
                    page: ::gen_models::traits::PageRequest::unbounded(),
                }
            }

            pub fn with_ref(mut self, history_ref: impl ::core::convert::Into<::std::string::String>) -> Self {
                self.history_ref = ::core::option::Option::Some(history_ref.into());
                self
            }

            #(#filter_methods)*

            pub fn order_by(
                mut self,
                field: #selector_field,
                direction: ::gen_models::traits::Direction,
            ) -> Self {
                self.order_by.push(::gen_models::traits::SqlOrder::new(
                    field.as_sql(),
                    direction,
                ));
                self
            }

            pub fn limit(mut self, limit: u32) -> Self {
                self.page.limit = ::core::option::Option::Some(limit);
                self
            }

            pub fn offset(mut self, offset: u32) -> Self {
                self.page.offset = offset;
                self
            }

            pub fn load(self) -> ::std::vec::Vec<#model> {
                <#model as ::gen_models::traits::QuerySelect>::select(self.conn, &self)
            }

            pub(crate) fn push_filter(
                mut self,
                filter: ::gen_models::traits::SqlFilter,
            ) -> Self {
                self.filters.push(filter);
                self
            }
        }

        impl ::gen_models::traits::ModelSelect for #selector<'_> {
            fn source_clause(&self) -> ::std::string::String {
                #source_clause
            }

            fn source_params(&self) -> ::std::vec::Vec<::gen_models::traits::SqlValue> {
                self.history_ref
                    .as_ref()
                    .map(|history_ref| {
                        ::std::vec![::gen_models::traits::SqlValue::from(history_ref.clone())]
                    })
                    .unwrap_or_default()
            }

            fn select_clause(&self) -> &'static str {
                #select_clause
            }

            fn filters(&self) -> &[::gen_models::traits::SqlFilter] {
                &self.filters
            }

            fn order_by(&self) -> &[::gen_models::traits::SqlOrder] {
                &self.order_by
            }

            fn page(&self) -> ::gen_models::traits::PageRequest {
                self.page
            }
        }

        impl #model {
            pub fn select(
                conn: &::gen_models::traits::SelectConnection,
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
                self.filters.push(::gen_models::traits::SqlFilter::new(
                    ::std::concat!(#column, " = ?"),
                    ::std::vec![::gen_models::traits::sql_value(&value)],
                ));
                self
            }
        }
    } else {
        quote! {
            pub fn #field(mut self, value: #value_type) -> Self {
                self.filters.push(::gen_models::traits::SqlFilter::new(
                    ::std::concat!(#column, " = ?"),
                    ::std::vec![::gen_models::traits::sql_value(&value)],
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
                self.filters.push(::gen_models::traits::SqlFilter::new(
                    ::std::concat!("instr(lower(", #column, "), lower(?)) > 0"),
                    ::std::vec![::gen_models::traits::sql_value(&value)],
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
                self.filters.push(::gen_models::traits::SqlFilter::new(
                    ::std::concat!(#column, " IS NULL"),
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
