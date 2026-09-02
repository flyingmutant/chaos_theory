// Copyright 2026 Gregory Petrosyan <pgregory@pgregory.net>
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Proc-macro implementation for `#[derive(chaos_theory::Arbitrary)]`.

extern crate alloc;
extern crate proc_macro;

use alloc::collections::BTreeSet;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, Data, DataEnum, DataStruct, DeriveInput, Error, Fields, FieldsNamed, FieldsUnnamed,
    GenericParam, Generics, Type, Variant, parse_macro_input, spanned::Spanned as _,
    visit::Visit as _,
};

const ATTR_NAMESPACE: &str = "chaos_theory";

/// Derives an implementation of `chaos_theory::Arbitrary`.
///
/// The derive supports structs and non-empty enums; unions are not supported. By default, it
/// generates every field with that field type's `Arbitrary` implementation.
///
/// The derived type must implement `Debug` (usually by deriving it alongside `Arbitrary`). Type
/// parameters used by default-generated fields receive an `Arbitrary` bound automatically. A
/// type parameter used only by fields with a custom generator does not receive that bound.
///
/// # Generator modifiers
///
/// `#[chaos_theory(...)]` accepts two modifiers:
///
/// - `generator = EXPR` replaces a field's default generator. The expression must implement
///   `Generator` with the field's type as its item. This modifier is only supported on fields.
/// - `filter = EXPR` filters generated values with a predicate of the form `Fn(&T) -> bool`. It
///   can be applied to a field or to the derived struct or enum, but not to an enum variant.
///
/// Put `generator` and `filter` in separate attributes when using both on one field. The custom
/// generator is applied before the filter, regardless of attribute order. Filters that reject
/// too many values can make generation fail after exhausting its retries.
///
/// ```ignore
/// #[derive(Debug, chaos_theory::Arbitrary)]
/// #[chaos_theory(filter = |header| header.length >= header.alignment)]
/// struct Header {
///     length: u16,
///     #[chaos_theory(generator = chaos_theory::make::int_in(1..=255))]
///     #[chaos_theory(filter = |alignment| alignment.is_power_of_two())]
///     alignment: u16,
/// }
/// ```
#[proc_macro_derive(Arbitrary, attributes(chaos_theory))]
pub fn derive_arbitrary(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_derive_arbitrary(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn expand_derive_arbitrary(input: DeriveInput) -> syn::Result<TokenStream> {
    let container_config = parse_container_config(&input.attrs)?;
    ensure_supported_data_attributes(&input.data)?;

    let DeriveInput {
        ident: name,
        generics,
        data,
        ..
    } = input;
    let generics = add_arbitrary_bounds(generics, &data)?;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let body = match &data {
        Data::Struct(data) => derive_struct_body(data)?,
        Data::Enum(data) => derive_enum_body(&name, data)?,
        Data::Union(data) => {
            return Err(Error::new(
                data.union_token.span(),
                "`Arbitrary` derive does not support unions",
            ));
        }
    };
    let generator = apply_generator_config(
        &container_config,
        quote! {
            ::chaos_theory::make::from_next(
                |src: &mut ::chaos_theory::SourceEx, example: Option<&Self>| {
                    #body
                },
            )
        },
    );

    Ok(quote! {
        impl #impl_generics ::chaos_theory::Arbitrary for #name #ty_generics #where_clause {
            fn arbitrary() -> impl ::chaos_theory::Generator<Item = Self> {
                #generator
            }
        }
    })
}

fn ensure_supported_data_attributes(data: &Data) -> syn::Result<()> {
    match data {
        Data::Struct(DataStruct { fields, .. }) => {
            for field in fields {
                parse_field_config(field)?;
            }
        }
        Data::Enum(DataEnum { variants, .. }) => {
            for variant in variants {
                for attr in &variant.attrs {
                    ensure_no_variant_attribute(attr)?;
                }
                for field in &variant.fields {
                    parse_field_config(field)?;
                }
            }
        }
        Data::Union(_) => {}
    }

    Ok(())
}

fn ensure_no_variant_attribute(attr: &Attribute) -> syn::Result<()> {
    if attr.path().is_ident(ATTR_NAMESPACE) {
        return Err(Error::new_spanned(
            attr,
            "`#[chaos_theory(...)]` attributes are not supported on enum variants",
        ));
    }
    Ok(())
}

#[derive(Default)]
struct GeneratorConfig {
    generator: Option<TokenStream>,
    filter: Option<TokenStream>,
}

struct RawGeneratorModifier {
    name: syn::Ident,
    value: TokenStream,
}

impl syn::parse::Parse for RawGeneratorModifier {
    fn parse(input: syn::parse::ParseStream<'_>) -> syn::Result<Self> {
        let name = input.parse()?;
        input.parse::<syn::Token![=]>()?;
        let value = input.parse()?;
        Ok(Self { name, value })
    }
}

fn parse_container_config(attrs: &[Attribute]) -> syn::Result<GeneratorConfig> {
    parse_generator_config(attrs, false)
}

fn parse_field_config(field: &syn::Field) -> syn::Result<GeneratorConfig> {
    parse_generator_config(&field.attrs, true)
}

fn parse_generator_config(
    attrs: &[Attribute],
    allow_generator: bool,
) -> syn::Result<GeneratorConfig> {
    let mut config = GeneratorConfig::default();

    for attr in attrs {
        if !attr.path().is_ident(ATTR_NAMESPACE) {
            continue;
        }

        let syn::Meta::List(list) = &attr.meta else {
            return Err(Error::new_spanned(
                attr,
                "expected `#[chaos_theory(MODIFIER = EXPR)]`",
            ));
        };
        let modifier = syn::parse2::<RawGeneratorModifier>(list.tokens.clone())
            .map_err(|_| Error::new_spanned(attr, "expected `#[chaos_theory(MODIFIER = EXPR)]`"))?;
        if modifier.value.is_empty() {
            return Err(Error::new_spanned(
                attr,
                "expected `#[chaos_theory(MODIFIER = EXPR)]`",
            ));
        }

        let slot = match modifier.name.to_string().as_str() {
            "generator" if allow_generator => &mut config.generator,
            "generator" => {
                return Err(Error::new_spanned(
                    modifier.name,
                    "`generator` modifier is only supported on fields",
                ));
            }
            "filter" => &mut config.filter,
            _ => {
                let expected = if allow_generator {
                    "expected `generator` or `filter`"
                } else {
                    "expected `filter`"
                };
                return Err(Error::new_spanned(modifier.name, expected));
            }
        };
        if slot.is_some() {
            let name = modifier.name;
            return Err(Error::new_spanned(
                &name,
                format!("duplicate `{name}` modifier"),
            ));
        }
        *slot = Some(modifier.value);
    }

    Ok(config)
}

fn apply_generator_config(config: &GeneratorConfig, generator: TokenStream) -> TokenStream {
    if let Some(filter) = &config.filter {
        quote! { ::chaos_theory::Generator::filter(#generator, #filter) }
    } else {
        generator
    }
}

fn add_arbitrary_bounds(mut generics: Generics, data: &Data) -> syn::Result<Generics> {
    let type_params = generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(type_param) => Some(type_param.ident.to_string()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let used_params = used_type_params(data, &type_params)?;

    for param in &mut generics.params {
        if let GenericParam::Type(type_param) = param
            && used_params.contains(&type_param.ident.to_string())
        {
            let bound = syn::parse_str("::chaos_theory::Arbitrary")
                .expect("internal error: invalid Arbitrary bound path");
            type_param.bounds.push(bound);
        }
    }

    Ok(generics)
}

fn used_type_params(data: &Data, type_params: &BTreeSet<String>) -> syn::Result<BTreeSet<String>> {
    let mut visitor = UsedTypeParams {
        type_params,
        used: BTreeSet::new(),
    };

    match data {
        Data::Struct(data) => {
            for field in &data.fields {
                if parse_field_config(field)?.generator.is_none() {
                    visitor.visit_type(&field.ty);
                }
            }
        }
        Data::Enum(data) => {
            for variant in &data.variants {
                for field in &variant.fields {
                    if parse_field_config(field)?.generator.is_none() {
                        visitor.visit_type(&field.ty);
                    }
                }
            }
        }
        Data::Union(_) => {}
    }

    Ok(visitor.used)
}

struct UsedTypeParams<'a> {
    type_params: &'a BTreeSet<String>,
    used: BTreeSet<String>,
}

impl<'ast> syn::visit::Visit<'ast> for UsedTypeParams<'_> {
    fn visit_type_path(&mut self, type_path: &'ast syn::TypePath) {
        if type_path.qself.is_none() && is_phantom_data_path(&type_path.path) {
            return;
        }

        if type_path.qself.is_none()
            && type_path.path.leading_colon.is_none()
            && type_path.path.segments.len() == 1
        {
            let ident = type_path.path.segments[0].ident.to_string();
            if self.type_params.contains(&ident) {
                self.used.insert(ident);
                return;
            }
        }

        syn::visit::visit_type_path(self, type_path);
    }
}

fn is_phantom_data_path(path: &syn::Path) -> bool {
    path.segments
        .last()
        .is_some_and(|segment| segment.ident == "PhantomData")
}

fn derive_struct_body(data: &DataStruct) -> syn::Result<TokenStream> {
    match &data.fields {
        Fields::Named(fields) => derive_named_struct_ctor(fields),
        Fields::Unnamed(fields) => derive_unnamed_struct_ctor(fields),
        Fields::Unit => Ok(quote!(Self)),
    }
}

fn derive_named_struct_ctor(fields: &FieldsNamed) -> syn::Result<TokenStream> {
    let field_exprs = fields
        .named
        .iter()
        .map(|field| {
            let config = parse_field_config(field)?;
            let Some(field_ident) = field.ident.as_ref() else {
                unreachable!("internal error: named field without ident");
            };
            let label = field_ident.to_string();
            let example = quote!(example.map(|e| &e.#field_ident));
            let field_value = field_generator_expr(&config, &field.ty, &label, &example);
            Ok(quote! { #field_ident: #field_value })
        })
        .collect::<syn::Result<Vec<_>>>()?;

    Ok(quote! {
        Self {
            #(#field_exprs,)*
        }
    })
}

fn derive_unnamed_struct_ctor(fields: &FieldsUnnamed) -> syn::Result<TokenStream> {
    let field_exprs = fields
        .unnamed
        .iter()
        .enumerate()
        .map(|(ix, field)| {
            let config = parse_field_config(field)?;
            let field_ix = syn::Index::from(ix);
            let label = ix.to_string();
            let example = quote!(example.map(|e| &e.#field_ix));
            Ok(field_generator_expr(&config, &field.ty, &label, &example))
        })
        .collect::<syn::Result<Vec<_>>>()?;

    Ok(quote! {
        Self(
            #(#field_exprs,)*
        )
    })
}

fn field_generator_expr(
    config: &GeneratorConfig,
    field_type: &Type,
    label: &str,
    example: &TokenStream,
) -> TokenStream {
    if config.generator.is_none() && config.filter.is_none() {
        return quote! { src.any(#label, #example) };
    }

    let generator = config.generator.as_ref().map_or_else(
        || quote! { <#field_type as ::chaos_theory::Arbitrary>::arbitrary() },
        |generator| quote! { #generator },
    );
    let generator = apply_generator_config(config, generator);
    quote! { src.any_of(#label, #generator, #example) }
}

fn derive_enum_body(type_ident: &syn::Ident, data: &DataEnum) -> syn::Result<TokenStream> {
    if data.variants.is_empty() {
        return Err(Error::new_spanned(
            type_ident,
            "`Arbitrary` derive requires enums to have at least one variant",
        ));
    }

    let type_label = format!("<{type_ident}>");

    let example_index_arms = data
        .variants
        .iter()
        .enumerate()
        .map(|(ix, variant)| {
            let variant_pat = enum_variant_pattern_for_example_index(variant);
            quote! { #variant_pat => #ix, }
        })
        .collect::<Vec<_>>();

    let variant_labels = data
        .variants
        .iter()
        .map(|variant| variant.ident.to_string())
        .collect::<Vec<_>>();

    let variant_bodies = data
        .variants
        .iter()
        .enumerate()
        .map(|(ix, variant)| {
            let body = derive_enum_variant_ctor(variant)?;
            Ok(quote! { #ix => #body, })
        })
        .collect::<syn::Result<Vec<_>>>()?;

    Ok(quote! {
        let example_index = example.map(|e| match e {
            #(#example_index_arms)*
        });

        let variants = [#(#variant_labels,)*];
        let variants_num = ::core::num::NonZero::new(variants.len())
            .expect("internal error: no variants");

        src.select(
            #type_label,
            example_index,
            variants_num,
            |ix| variants[ix],
            |src, _variant, ix| match ix {
                #(#variant_bodies)*
                _ => unreachable!(),
            },
        )
    })
}

fn enum_variant_pattern_for_example_index(variant: &Variant) -> TokenStream {
    let variant_ident = &variant.ident;
    match &variant.fields {
        Fields::Named(_) => quote!(Self::#variant_ident { .. }),
        Fields::Unnamed(_) => quote!(Self::#variant_ident(..)),
        Fields::Unit => quote!(Self::#variant_ident),
    }
}

fn derive_enum_variant_ctor(variant: &Variant) -> syn::Result<TokenStream> {
    let variant_ident = &variant.ident;
    match &variant.fields {
        Fields::Named(fields) => {
            let field_exprs = fields
                .named
                .iter()
                .map(|field| {
                    let config = parse_field_config(field)?;
                    let Some(field_ident) = field.ident.as_ref() else {
                        unreachable!("internal error: named field without ident");
                    };
                    let label = field_ident.to_string();
                    let example = quote! {
                        match example {
                            Some(Self::#variant_ident { #field_ident, .. }) => Some(#field_ident),
                            _ => None,
                        }
                    };
                    let value = field_generator_expr(&config, &field.ty, &label, &example);
                    Ok(quote! {
                        #field_ident: #value
                    })
                })
                .collect::<syn::Result<Vec<_>>>()?;

            Ok(quote! {
                Self::#variant_ident {
                    #(#field_exprs,)*
                }
            })
        }
        Fields::Unnamed(fields) => {
            let field_exprs = fields
                .unnamed
                .iter()
                .enumerate()
                .map(|(ix, field)| {
                    let config = parse_field_config(field)?;
                    let label = ix.to_string();
                    let bindings = (0..fields.unnamed.len())
                        .map(|bind_ix| format_ident!("__example_{bind_ix}"))
                        .collect::<Vec<_>>();
                    let selected = &bindings[ix];
                    let example = quote! {
                        match example {
                            Some(Self::#variant_ident(#(#bindings),*)) => Some(#selected),
                            _ => None,
                        }
                    };
                    Ok(field_generator_expr(&config, &field.ty, &label, &example))
                })
                .collect::<syn::Result<Vec<_>>>()?;

            Ok(quote! {
                Self::#variant_ident(
                    #(#field_exprs,)*
                )
            })
        }
        Fields::Unit => Ok(quote!(Self::#variant_ident)),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        env, fs,
        path::{Path, PathBuf},
    };

    use syn::{Data, DataEnum, DataStruct, DeriveInput, Item, Token, punctuated::Punctuated};

    use super::expand_derive_arbitrary;

    const BLESS_ENV: &str = "BLESS";

    #[test]
    fn golden_shared_cases() {
        let actual = render_shared_cases();
        let golden = golden_path();

        if env::var_os(BLESS_ENV).is_some() {
            if let Some(parent) = golden.parent() {
                fs::create_dir_all(parent).unwrap_or_else(|err| {
                    panic!(
                        "failed to create golden directory '{}': {err}",
                        parent.display()
                    )
                });
            }
            fs::write(&golden, &actual).unwrap_or_else(|err| {
                panic!("failed to write golden file '{}': {err}", golden.display())
            });
        }

        let expected = fs::read_to_string(&golden).unwrap_or_else(|err| {
            panic!(
                "failed to read golden file '{}': {err}. Run with {BLESS_ENV}=1 to create it.",
                golden.display()
            )
        });

        assert_eq!(actual, expected);
    }

    fn render_shared_cases() -> String {
        let input_path = shared_cases_path();
        let input = fs::read_to_string(&input_path).unwrap_or_else(|err| {
            panic!(
                "failed to read shared derive cases '{}': {err}",
                input_path.display()
            )
        });

        let file = syn::parse_file(&input).unwrap_or_else(|err| {
            panic!(
                "failed to parse shared derive cases '{}': {err}",
                input_path.display()
            )
        });

        let inputs = derive_inputs(&file);
        let mut out = String::new();
        out.push_str("// This file is generated by chaos_theory_derive golden tests.\n");
        out.push_str("// Do not edit manually.\n");
        out.push_str("// To regenerate: BLESS=1 cargo test -p chaos_theory_derive\n\n");
        for input in inputs {
            let name = input.ident.to_string();
            let expanded = expand_derive_arbitrary(input)
                .unwrap_or_else(|err| panic!("failed to expand derive for '{name}': {err}"));
            let item = syn::parse2::<syn::Item>(expanded)
                .unwrap_or_else(|err| panic!("failed to parse expanded item for '{name}': {err}"));
            let pretty = prettyplease::unparse(&syn::File {
                shebang: None,
                frontmatter: None,
                attrs: Vec::new(),
                items: vec![item],
            });
            out.push_str("// === ");
            out.push_str(&name);
            out.push_str(" ===\n");
            out.push_str(&pretty);
            out.push('\n');
        }
        out
    }

    fn derive_inputs(file: &syn::File) -> Vec<DeriveInput> {
        file.items
            .iter()
            .filter_map(item_to_derive_input)
            .filter(|input| has_derive_arbitrary(&input.attrs))
            .collect()
    }

    fn item_to_derive_input(item: &Item) -> Option<DeriveInput> {
        match item {
            Item::Struct(item) => Some(DeriveInput {
                attrs: item.attrs.clone(),
                vis: item.vis.clone(),
                ident: item.ident.clone(),
                generics: item.generics.clone(),
                data: Data::Struct(DataStruct {
                    struct_token: item.struct_token,
                    fields: item.fields.clone(),
                    semi_token: item.semi_token,
                }),
            }),
            Item::Enum(item) => Some(DeriveInput {
                attrs: item.attrs.clone(),
                vis: item.vis.clone(),
                ident: item.ident.clone(),
                generics: item.generics.clone(),
                data: Data::Enum(DataEnum {
                    enum_token: item.enum_token,
                    brace_token: item.brace_token,
                    variants: item.variants.clone(),
                }),
            }),
            _ => None,
        }
    }

    fn has_derive_arbitrary(attrs: &[syn::Attribute]) -> bool {
        attrs.iter().any(|attr| {
            if !attr.path().is_ident("derive") {
                return false;
            }
            if let syn::Meta::List(list) = &attr.meta {
                let parsed =
                    list.parse_args_with(Punctuated::<syn::Path, Token![,]>::parse_terminated);
                if let Ok(paths) = parsed {
                    return paths.iter().any(|path| {
                        path.is_ident("Arbitrary")
                            || path
                                .segments
                                .last()
                                .is_some_and(|segment| segment.ident == "Arbitrary")
                    });
                }
            }
            false
        })
    }

    fn shared_cases_path() -> PathBuf {
        workspace_root().join("testdata/derive_cases.rs")
    }

    fn golden_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/derive_cases.out.rs")
    }

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap_or_else(|| panic!("derive crate has no workspace parent"))
            .to_path_buf()
    }
}
