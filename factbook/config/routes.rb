Rails.application.routes.draw do
  resources :contents
  root "content#index"
  get "/content", to: "content#index"
  # Define your application routes per the DSL in https://guides.rubyonrails.org/routing.html

  # Defines the root path route ("/")
  # root "articles#index"
end
